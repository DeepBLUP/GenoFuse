import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
import torch.optim as optim
from timm.models.layers import DropPath, trunc_normal_
from torch.utils.data import Dataset 
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import numpy as np


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.5):  
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ConvBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6), drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(outplanes)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = norm_layer(outplanes)

        self.res_conv = res_conv
        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x

class FCUDown(nn.Module):
    def __init__(self, inplanes, outplanes, dw_stride=2, act_layer=nn.GELU,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super(FCUDown, self).__init__()
        self.dw_stride = max(1, dw_stride)  

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=self.dw_stride, stride=self.dw_stride)

        self.ln = norm_layer(outplanes)
        self.act = act_layer()


    def forward(self, x, x_t):
        x = self.conv_project(x)

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x

class FCUUp(nn.Module):
    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU,
                 norm_layer=partial(nn.BatchNorm2d, eps=1e-6)):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = norm_layer(outplanes)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))

class Med_ConvBlock(nn.Module):
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):
        super(Med_ConvBlock, self).__init__()

        expansion = 4
        med_planes = inplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(inplace=True)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=True)

        self.drop_block = drop_block
        self.drop_path = drop_path

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        if self.drop_block is not None:
            x = self.drop_block(x)
        x = self.act2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        if self.drop_block is not None:
            x = self.drop_block(x)

        if self.drop_path is not None:
            x = self.drop_path(x)

        x += residual
        x = self.act3(x)

        return x

class ConvTransBlock(nn.Module):
    def __init__(self, inplanes, outplanes, res_conv, stride, embed_dim, dw_stride=2, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 last_fusion=False, num_med_block=0, groups=1):
        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)

        if last_fusion:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, stride=2, res_conv=True, groups=groups)
        else:
            self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)

        if num_med_block > 0:
            self.med_block = []
            for i in range(num_med_block):
                self.med_block.append(Med_ConvBlock(inplanes=outplanes, groups=groups))
            self.med_block = nn.ModuleList(self.med_block)

        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=max(1, dw_stride))
        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=max(1, dw_stride))
        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

        self.dw_stride = max(1, dw_stride) 
        self.embed_dim = embed_dim
        self.num_med_block = num_med_block
        self.last_fusion = last_fusion


    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)
        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        if x_st.shape[1] != x_t.shape[1]:
            diff = abs(x_st.shape[1] - x_t.shape[1])
            if x_st.shape[1] > x_t.shape[1]:
                x_st = x_st[:, :x_t.shape[1], :]
            else:
                x_t = x_t[:, :x_st.shape[1], :]

        x_t = self.trans_block(x_st + x_t)

        if self.num_med_block > 0:
            for m in self.med_block:
                x = m(x)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t

class Conformer(nn.Module):
    def __init__(self, patch_size=16, in_chans=1, num_classes=1, base_channel=64, channel_ratio=4, num_med_block=0,
                 embed_dim=128, depth=6, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(int(256 * channel_ratio), num_classes)

        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.trans_dpr[0])

        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                                num_med_block=num_med_block
                            ))

        stage_2_channel = int(base_channel * channel_ratio * 2)
        init_stage = fin_stage
        fin_stage = fin_stage + depth // 3
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                                num_med_block=num_med_block))

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        init_stage = fin_stage
        fin_stage = fin_stage + depth // 3
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            last_fusion = True if i == depth else False
            self.add_module('conv_trans_' + str(i),
                            ConvTransBlock(
                                in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
                                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=self.trans_dpr[i-1],
                                num_med_block=num_med_block, last_fusion=last_fusion))
        self.fin_stage = fin_stage

        trunc_normal_(self.cls_token, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'cls_token'}

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x_base = self.maxpool(self.act1(self.bn1(self.conv1(x))))

        x = self.conv_1(x_base, return_x_2=False)

        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)
        
        for i in range(2, self.fin_stage):
            x, x_t = getattr(self, 'conv_trans_' + str(i))(x, x_t)

        x_p = self.pooling(x).flatten(1)
        conv_cls = self.conv_cls_head(x_p)

        x_t = self.trans_norm(x_t)
        tran_cls = self.trans_cls_head(x_t[:, 0])

        return [conv_cls, tran_cls]

class ChunkedDataset(Dataset):
    def __init__(self, filepath, chunk_size, scaler):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.scaler = scaler
        self.data_chunks = pd.read_csv(filepath, chunksize=chunk_size)
        self.chunk_list = list(self.data_chunks)
        self.total_size = sum([len(chunk) for chunk in self.chunk_list])

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        row_idx = idx % self.chunk_size
        chunk = self.chunk_list[chunk_idx]
        
        input_data = chunk.iloc[row_idx, 6:].values
        target_data = chunk.iloc[row_idx, 5] 
        input_data = self.scaler.transform([input_data])[0]
        
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32)


class ChunkedDataset(Dataset):
    def __init__(self, filepath, chunk_size, scaler):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.scaler = scaler
        self.data_chunks = pd.read_csv(filepath, chunksize=chunk_size)
        self.chunk_list = list(self.data_chunks)
        self.total_size = sum([len(chunk) for chunk in self.chunk_list])

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        chunk_idx = idx // self.chunk_size
        row_idx = idx % self.chunk_size
        chunk = self.chunk_list[chunk_idx]
        
        input_data = chunk.iloc[row_idx, 6:].values
        target_data = chunk.iloc[row_idx, 5]
        input_data = self.scaler.transform([input_data])[0]
        
        return torch.tensor(input_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.float32)


class EnsembleModel(nn.Module):
    def __init__(self, deep_model, rf_model, rf_weight=0.3, device='cpu'):
        super(EnsembleModel, self).__init__()
        self.deep_model = deep_model
        self.rf_model = rf_model
        self.rf_weight = rf_weight
        self.device = device

    def forward(self, x, x_rf):
        deep_output = self.deep_model(x)[0]

        rf_output = torch.tensor(self.rf_model.predict(x_rf), dtype=torch.float32).to(self.device).view(-1, 1)

        ensemble_output = (1 - self.rf_weight) * deep_output + self.rf_weight * rf_output

        return ensemble_output


data = pd.read_csv(r'testdate.csv', low_memory=False)
input_data = data.iloc[:, 6:].values
target_data = data.iloc[:, 5].values

scaler = StandardScaler()
scaler.fit(input_data)

chunk_size = 10
dataset = ChunkedDataset(r'testdate.csv', chunk_size, scaler)

print("Training RandomForestRegressor model...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(input_data, target_data)
print("RandomForestRegressor model training completed.")

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

deep_model = Conformer(patch_size=4, in_chans=1, num_classes=1, base_channel=64, channel_ratio=4,
                       embed_dim=64, depth=12, num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                       drop_rate=0.5, attn_drop_rate=0, drop_path_rate=0.1).to(device)

ensemble_model = EnsembleModel(deep_model, rf_model, rf_weight=0.3, device=device).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(deep_model.parameters(), lr=0.001, weight_decay=1e-2)


num_epochs = 20
best_pcc = -1
best_epoch = 0

for epoch in range(num_epochs):
    ensemble_model.train()
    running_loss = 0.0
    all_targets = []
    all_predictions = []

    print(f'Epoch {epoch+1}/{num_epochs}')
    train_loader = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False)

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.unsqueeze(1).unsqueeze(2)  
        inputs_rf = inputs.view(inputs.size(0), -1).cpu().numpy()  
        
        if targets.dim() == 1:
            targets = targets.view(-1, 1)

        optimizer.zero_grad()
        outputs = ensemble_model(inputs, inputs_rf)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        all_targets.extend(targets.cpu().numpy().flatten().tolist())
        all_predictions.extend(outputs.detach().cpu().numpy().flatten().tolist())

    epoch_loss = running_loss / len(train_dataloader.dataset)
    r2 = r2_score(all_targets, all_predictions)
    pcc, _ = pearsonr(all_targets, all_predictions)
    print(f'Epoch {epoch+1} Train - Loss: {epoch_loss:.4f}, R²: {r2:.4f}, PCC: {pcc:.4f}')

    
    ensemble_model.eval()
    val_targets = []
    val_predictions = []
    with torch.no_grad():
        for inputs, targets in val_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.unsqueeze(1).unsqueeze(2)
            inputs_rf = inputs.view(inputs.size(0), -1).cpu().numpy()

            outputs = ensemble_model(inputs, inputs_rf)
            val_targets.extend(targets.cpu().numpy().flatten().tolist())
            val_predictions.extend(outputs.cpu().numpy().flatten().tolist())

    val_r2 = r2_score(val_targets, val_predictions)
    val_pcc, _ = pearsonr(val_targets, val_predictions)
    print(f'Epoch {epoch+1} Validation - R²: {val_r2:.4f}, PCC: {val_pcc:.4f}')

    if val_pcc > best_pcc:
        best_pcc = val_pcc
        best_epoch = epoch
        torch.save(ensemble_model.state_dict(), 'best_ensemble_model.pth')

print(f'Best Validation PCC: {best_pcc:.4f} at epoch {best_epoch+1}')