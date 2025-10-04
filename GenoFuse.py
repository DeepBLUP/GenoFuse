import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import torch.optim as optim
from timm.models.layers import DropPath, trunc_normal_
from torch.utils.data import Dataset
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time
import matplotlib.pyplot as plt
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
import os
import logging
from datetime import datetime
import json
from sklearn.model_selection import KFold

# ==============================================================================
# 1. CONFIGURATION SECTION
# ==============================================================================

# --- File and Path Configuration ---
# Main training data paths
GENO_DATA_PATH = r'/path/to/your/genotype.raw'  # [REQUIRED] Path to genotype file (.raw format, space-separated)
PHENO_DATA_PATH = r'/path/to/your/phenotype.txt'  # [REQUIRED] Path to phenotype file (.txt format, tab-separated)

# (Optional) Independent test set paths
TEST_GENO_DATA_PATH = None     # [OPTIONAL] Path to independent test genotype file (.raw)
TEST_PHENO_DATA_PATH = None    # [OPTIONAL] Path to independent test phenotype file (.txt)

# Output directory
OUTPUT_DIR = None              # [OPTIONAL] Directory for all output files (logs, models, plots). If None, saves to phenotype file directory

# --- Data Matching and Selection Configuration ---
GENO_ID_COL = 'IID'           # Column name for individual ID in genotype file (.raw). Usually 'IID'
PHENO_ID_COL = 'ID'           # Column name for individual ID in phenotype file (.txt)
PHENO_TARGET_COL_IDX = 1      # Which column to use as training target in phenotype file (0-indexed)

# ==============================================================================
# 2. BASIC PARAMETER TUNING (Commonly Adjusted)
# ==============================================================================

# --- Training Parameters ---
SEED = 123                    # Random seed for reproducibility
BATCH_SIZE = 30               # Batch size for training (adjust based on GPU memory)
NUM_EPOCHS = 120              # Number of training epochs
GRADIENT_ACCUMULATION_STEPS = 6  # Gradient accumulation steps

# --- Learning Rate and Optimization ---
LEARNING_RATE = 0.002         # Initial learning rate
WEIGHT_DECAY = 0.003          # Weight decay for regularization

# --- Dropout Rates (Regularization) ---
DROP_RATE = 0.4               # General dropout rate
ATTN_DROP_RATE = 0.3          # Attention dropout rate
DROP_PATH_RATE = 0.2          # Drop path rate

# --- Cross Validation ---
USE_CROSS_VALIDATION = True   # Enable cross-validation
N_SPLITS = 5                  # Number of folds for cross-validation
SAVE_PER_FOLD_MODELS = True   # Save model for each fold
SAVE_PER_FOLD_PLOTS = True    # Save training plots for each fold

# --- Early Stopping ---
EARLY_STOPPING_PATIENCE = 12  # Patience for early stopping
EARLY_STOPPING_MIN_DELTA = 0.008  # Minimum change to qualify as improvement

# ==============================================================================
# 3. ADVANCED PARAMETER TUNING (Architecture & Technical)
# ==============================================================================

# --- Model Architecture Parameters ---
PATCH_SIZE = 4                # Patch size for Conformer model
BASE_CHANNEL = 16             # Base number of channels
EMBED_DIM = 24                # Embedding dimension
DEPTH = 3                     # Number of transformer layers
NUM_HEADS = 2                 # Number of attention heads

# --- Learning Rate Scheduler ---
SCHEDULER_T_0 = 10            # Initial restart period for CosineAnnealingWarmRestarts
SCHEDULER_T_MULT = 2          # Multiplication factor for restart period
SCHEDULER_ETA_MIN = 5e-6      # Minimum learning rate

# --- Random Forest Parameters ---
RF_N_ESTIMATORS = 50          # Number of trees in random forest
RF_MAX_DEPTH = 8              # Maximum depth of trees
RF_MAX_FEATURES = 'sqrt'      # Number of features to consider for splits
RF_MIN_SAMPLES_SPLIT = 10     # Minimum samples required to split node
RF_MIN_SAMPLES_LEAF = 5       # Minimum samples required at leaf node

# --- Memory Optimization ---
USE_MIXED_PRECISION = True    # Use FP16 mixed precision training
USE_GRADIENT_CHECKPOINTING = True  # Use gradient checkpointing to save memory
EMPTY_CACHE_FREQUENCY = 10    # Clear GPU cache every N batches
PIN_MEMORY = False            # Use pinned memory (disable for limited GPU memory)
NUM_WORKERS = 0               # Number of data loader workers (0 for limited memory)

# --- Multi-GPU Configuration ---
USE_MULTI_GPU = True          # Enable multi-GPU training
MULTI_GPU_STRATEGY = "DataParallel"  # Multi-GPU strategy: "DataParallel" or "DistributedDataParallel"
GPU_IDS = [0, 1, 2]           # GPU IDs to use for training

# --- Data Processing ---
OUTLIER_THRESHOLD = 3         # Threshold for outlier detection (standard deviations)
TRAIN_SPLIT_RATIO = 0.7       # Training set ratio (when not using cross-validation)
VALIDATION_SPLIT_RATIO = 0.15 # Validation set ratio (when not using cross-validation)
TRAIN_VALIDATION_SPLIT_RATIO = 0.8  # Train/validation split when using independent test set

# --- Feature Analysis ---
FEATURE_IMPORTANCE_TOP_N = 10 # Number of top features to display in analysis

# ==============================================================================
# 4. DERIVED CONFIGURATIONS (Auto-computed)
# ==============================================================================

# Device configuration with multi-GPU support
if torch.cuda.is_available() and USE_MULTI_GPU:
    available_gpus = torch.cuda.device_count()
    if GPU_IDS is None:
        GPU_IDS = list(range(available_gpus))
    else:
        GPU_IDS = [gpu_id for gpu_id in GPU_IDS if gpu_id < available_gpus]

    if len(GPU_IDS) > 1:
        DEVICE = f"cuda:{GPU_IDS[0]}"  # Primary GPU
        print(f"Detected {available_gpus} GPUs, using GPUs: {GPU_IDS}")
    else:
        DEVICE = "cuda:0"
        USE_MULTI_GPU = False
        print("Only one GPU available, using single GPU mode")
else:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    USE_MULTI_GPU = False

# Adjust batch size for single GPU
if not USE_MULTI_GPU:
    BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 8

# Conformer model parameters dictionary
CONFORMER_PARAMS = {
    'patch_size': PATCH_SIZE,
    'base_channel': BASE_CHANNEL,
    'embed_dim': EMBED_DIM,
    'depth': DEPTH,
    'num_heads': NUM_HEADS,
    'drop_rate': DROP_RATE,
    'attn_drop_rate': ATTN_DROP_RATE,
    'drop_path_rate': DROP_PATH_RATE,
}

# Optimizer parameters
OPTIMIZER_PARAMS = {
    'lr': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'betas': (0.9, 0.999)
}

# Loss function
CRITERION = nn.MSELoss()

# Scheduler parameters
SCHEDULER_PARAMS = {
    'T_0': SCHEDULER_T_0,
    'T_mult': SCHEDULER_T_MULT,
    'eta_min': SCHEDULER_ETA_MIN
}

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': RF_N_ESTIMATORS,
    'max_depth': RF_MAX_DEPTH,
    'n_jobs': -1,
    'random_state': SEED,
    'max_features': RF_MAX_FEATURES,
    'min_samples_split': RF_MIN_SAMPLES_SPLIT,
    'min_samples_leaf': RF_MIN_SAMPLES_LEAF,
}

# Early stopping parameters
EARLY_STOPPING_PARAMS = {
    'patience': EARLY_STOPPING_PATIENCE,
    'warmup': 12,
    'min_delta': EARLY_STOPPING_MIN_DELTA,
    'monitor': 'loss',
    'restore_best_weights': True
}

# Dynamic RF weight adjustment
DYNAMIC_RF_WEIGHT = True

def setup_logging(log_path):
    """Configure logging to output to both console and file"""
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

def save_hyperparameters(output_dir, target_phenotype, timestamp):
    """Save hyperparameters to independent JSON file"""
    hyperparameters = {
        'training_session': f"{target_phenotype}_{timestamp}",
        'timestamp': timestamp,
        'target_phenotype': target_phenotype,
        'device_config': {
            'device': DEVICE,
            'use_multi_gpu': USE_MULTI_GPU,
            'gpu_ids': GPU_IDS if USE_MULTI_GPU else None,
            'multi_gpu_strategy': MULTI_GPU_STRATEGY if USE_MULTI_GPU else None
        },
        'training_params': {
            'batch_size': BATCH_SIZE,
            'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
            'num_epochs': NUM_EPOCHS,
            'seed': SEED
        },
        'cross_validation_params': {
            'use_cross_validation': USE_CROSS_VALIDATION,
            'n_splits': N_SPLITS if USE_CROSS_VALIDATION else None,
            'save_per_fold_models': SAVE_PER_FOLD_MODELS,
            'save_per_fold_plots': SAVE_PER_FOLD_PLOTS
        },
        'model_params': {
            'conformer': CONFORMER_PARAMS,
            'random_forest': RF_PARAMS
        },
        'optimizer_params': OPTIMIZER_PARAMS,
        'scheduler_params': SCHEDULER_PARAMS,
        'early_stopping_params': EARLY_STOPPING_PARAMS,
        'data_params': {
            'train_split_ratio': TRAIN_SPLIT_RATIO,
            'validation_split_ratio': VALIDATION_SPLIT_RATIO,
            'train_validation_split_ratio': TRAIN_VALIDATION_SPLIT_RATIO,
            'outlier_threshold': OUTLIER_THRESHOLD
        },
        'optimization_config': {
            'use_mixed_precision': USE_MIXED_PRECISION,
            'use_gradient_checkpointing': USE_GRADIENT_CHECKPOINTING,
            'empty_cache_frequency': EMPTY_CACHE_FREQUENCY,
            'pin_memory': PIN_MEMORY,
            'num_workers': NUM_WORKERS,
            'dynamic_rf_weight': DYNAMIC_RF_WEIGHT
        }
    }

    hyperparameter_path = os.path.join(output_dir, 'hyperparameters.json')
    with open(hyperparameter_path, 'w', encoding='utf-8') as f:
        json.dump(hyperparameters, f, indent=2, ensure_ascii=False)

    logging.info(f"Hyperparameters saved to: {hyperparameter_path}")
    return hyperparameter_path

def handle_outliers(data, columns, threshold=3):
    """Clip outliers based on standard deviation threshold"""
    data_clean = data.copy()
    for col in columns:
        mean_val = data_clean[col].mean()
        std_val = data_clean[col].std()
        data_clean[col] = data_clean[col].clip(
            lower=mean_val - threshold * std_val,
            upper=mean_val + threshold * std_val
        )
    return data_clean

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.3):
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
        self.act = act_layer(inplace=False)

    def forward(self, x, H, W):
        B, _, C = x.shape
        x_r = x[:, 1:].transpose(1, 2).contiguous().reshape(B, C, H, W)
        x_r = self.conv_project(x_r)
        x_r = self.bn(x_r)
        x_r = self.act(x_r)
        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))

class Med_ConvBlock(nn.Module):
    def __init__(self, inplanes, act_layer=nn.ReLU, groups=1, norm_layer=partial(nn.BatchNorm2d, eps=1e-6),
                 drop_block=None, drop_path=None):
        super(Med_ConvBlock, self).__init__()
        expansion = 4
        med_planes = inplanes // expansion
        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(med_planes)
        self.act1 = act_layer(inplace=False)
        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=1, groups=groups, padding=1, bias=False)
        self.bn2 = norm_layer(med_planes)
        self.act2 = act_layer(inplace=False)
        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(inplanes)
        self.act3 = act_layer(inplace=False)
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
            self.med_block = [Med_ConvBlock(inplanes=outplanes, groups=groups) for _ in range(num_med_block)]
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
             drop_rate=0.5, attn_drop_rate=0.3, drop_path_rate=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.trans_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.trans_norm = nn.LayerNorm(embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        self.conv_cls_head = nn.Linear(stage_3_channel, num_classes)

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

class EnsembleModel(nn.Module):
    def __init__(self, deep_model, rf_model, rf_weight=0.15, device='cpu'):
        super(EnsembleModel, self).__init__()
        self.deep_model = deep_model
        self.rf_model = rf_model
        self.rf_weight = rf_weight
        self.device = device
        self.is_multi_gpu = isinstance(deep_model, nn.DataParallel)

    def forward(self, x, x_rf):
        deep_outputs = self.deep_model(x)
        deep_output = deep_outputs[0] if isinstance(deep_outputs, list) else deep_outputs

        target_device = deep_output.device if self.is_multi_gpu else self.device

        with torch.cuda.device(target_device):
            rf_output = torch.tensor(
                self.rf_model.predict(x_rf),
                dtype=torch.float32,
                device=target_device
            ).view(-1, 1)

        ensemble_output = (1 - self.rf_weight) * deep_output + self.rf_weight * rf_output
        return ensemble_output

class EarlyStopping:
    def __init__(self, save_path, patience=10, warmup=5, min_delta=0.01,
                 smoothing=0.1, monitor='both', restore_best_weights=True, save_model=True):
        self.save_path = save_path
        self.patience = patience
        self.warmup = warmup
        self.min_delta = min_delta
        self.smoothing = smoothing
        self.monitor = monitor
        self.restore_best_weights = restore_best_weights
        self.save_model = save_model

        self.counter = 0
        self.best_loss = float('inf')
        self.best_pcc = -1
        self.best_r2 = -1
        self.best_mse = float('inf')
        self.best_mae = float('inf')
        self.best_epoch = 0
        self.best_weights = None
        self.smoothed_pcc = None
        self.early_stop = False

    def __call__(self, val_loss, val_pcc, val_r2, val_mse, val_mae, epoch, model):
        if epoch < self.warmup:
            return False

        self.smoothed_pcc = (
            val_pcc if self.smoothed_pcc is None
            else self.smoothing * val_pcc + (1 - self.smoothing) * self.smoothed_pcc
        )

        is_best = False
        if self.monitor == 'loss':
            if val_loss < self.best_loss - self.min_delta:
                is_best = True
        elif self.monitor == 'pcc':
            if self.smoothed_pcc > self.best_pcc + self.min_delta:
                is_best = True
        else:
            if val_loss < self.best_loss - self.min_delta or self.smoothed_pcc > self.best_pcc + self.min_delta:
                is_best = True

        if is_best:
            self.best_loss = min(val_loss, self.best_loss)
            self.best_pcc = max(self.smoothed_pcc, self.best_pcc)
            self.best_r2, self.best_mse, self.best_mae = val_r2, val_mse, val_mae
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())
                if self.save_model:
                    torch.save(model.state_dict(), self.save_path)
            logging.info(f"🎯 Best model updated (Epoch {epoch+1}) - Loss: {val_loss:.4f}, PCC: {val_pcc:.4f}, R²: {val_r2:.4f}")
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logging.info(f"\n🛑 Early stopping triggered! No improvement for {self.patience} consecutive epochs")
                logging.info(f"   Final best model from Epoch {self.best_epoch+1}")
                logging.info(f"   Validation Loss: {self.best_loss:.4f}, PCC: {self.best_pcc:.4f}, R²: {self.best_r2:.4f}")

        return self.early_stop

    def restore_weights(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logging.info(f"✅ Restored best model weights from Epoch {self.best_epoch+1}")

def train_model(model, train_dataloader, val_dataloader, test_dataloader,
                criterion, optimizer, scheduler, device, num_epochs, rf_model,
                dynamic_rf_weight, early_stopping_params, history_plot_path,
                accumulation_steps=1, save_history_plot=True):

    early_stopping = EarlyStopping(**early_stopping_params)
    history = {'train_loss': [], 'val_loss': [], 'train_r2': [], 'val_r2': [], 'train_pcc': [], 'val_pcc': [],
               'train_mse': [], 'val_mse': [], 'train_mae': [], 'val_mae': []}

    rf_weight = 0.20
    ensemble_model = EnsembleModel(model, rf_model, rf_weight=rf_weight, device=device).to(device)

    logging.info("Starting training...")
    start_time = time.time()
    scaler = GradScaler() if USE_MIXED_PRECISION and "cuda" in device else None
    use_amp = scaler is not None

    if use_amp: logging.info("✅ Mixed precision training enabled (FP16)")

    for epoch in range(num_epochs):
        ensemble_model.train()
        running_loss = 0.0
        all_targets, all_predictions = [], []
        train_loader = tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}/{num_epochs}", leave=False)

        optimizer.zero_grad()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
            inputs = inputs.unsqueeze(1).unsqueeze(2)
            inputs_rf = inputs.view(inputs.size(0), -1).cpu().numpy()

            if use_amp:
                with autocast(device_type='cuda'):
                    outputs = ensemble_model(inputs, inputs_rf)
                    loss = criterion(outputs, targets) / accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs = ensemble_model(inputs, inputs_rf)
                loss = criterion(outputs, targets) / accumulation_steps
                loss.backward()

            if (i + 1) % accumulation_steps == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

            running_loss += loss.item() * inputs.size(0) * accumulation_steps
            all_targets.extend(targets.cpu().numpy().flatten().tolist())
            all_predictions.extend(outputs.detach().cpu().numpy().flatten().tolist())
            if (i + 1) % EMPTY_CACHE_FREQUENCY == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Record training metrics
        epoch_loss = running_loss / len(train_dataloader.dataset)
        history['train_loss'].append(epoch_loss)
        r2, pcc, mse, mae = r2_score(all_targets, all_predictions), spearmanr(all_targets, all_predictions)[0], \
                            mean_squared_error(all_targets, all_predictions), mean_absolute_error(all_targets, all_predictions)
        history['train_r2'].append(r2); history['train_pcc'].append(pcc); history['train_mse'].append(mse); history['train_mae'].append(mae)
        logging.info(f'Epoch {epoch+1} Training - Loss: {epoch_loss:.4f}, R²: {r2:.4f}, PCC: {pcc:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}')

        # Record validation metrics
        val_metrics = evaluate_model(ensemble_model, val_dataloader, criterion, device, is_eval=True)
        for k, v in val_metrics.items():
            if f'val_{k}' in history: history[f'val_{k}'].append(v)
        logging.info(f"Epoch {epoch+1} Validation - Loss: {val_metrics['loss']:.4f}, R²: {val_metrics['r2']:.4f}, PCC: {val_metrics['pcc']:.4f}, MSE: {val_metrics['mse']:.4f}, MAE: {val_metrics['mae']:.4f}")

        scheduler.step(val_metrics['loss'])
        
        if dynamic_rf_weight and epoch > 8 and len(history['val_pcc']) >= 3:
            recent_trend = np.mean(history['val_pcc'][-3:]) - np.mean(history['val_pcc'][-6:-3]) if len(history['val_pcc']) >= 6 else 0
            current_pcc = history['val_pcc'][-1]
            if current_pcc > 0.4 and recent_trend > 0.008: rf_weight = max(0.12, rf_weight - 0.003)
            elif current_pcc < 0.3 or recent_trend < -0.015: rf_weight = min(0.25, rf_weight + 0.005)
            ensemble_model.rf_weight = rf_weight
        
        if early_stopping(val_metrics['loss'], val_metrics['pcc'], val_metrics['r2'], val_metrics['mse'], val_metrics['mae'], epoch, ensemble_model):
            break

    early_stopping.restore_weights(ensemble_model)
    logging.info(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    test_metrics = evaluate_model(ensemble_model, test_dataloader, criterion, device)
    logging.info(f"Test Set - Loss: {test_metrics['loss']:.4f}, R²: {test_metrics['r2']:.4f}, PCC: {test_metrics['pcc']:.4f}, MSE: {test_metrics['mse']:.4f}, MAE: {test_metrics['mae']:.4f}")

    if save_history_plot:
        plot_training_history(history, test_metrics, early_stopping.best_epoch + 1, history_plot_path)

    return ensemble_model, {
        'best_epoch': early_stopping.best_epoch, 'best_pcc': early_stopping.best_pcc,
        'best_r2': early_stopping.best_r2, 'best_mse': early_stopping.best_mse,
        'best_mae': early_stopping.best_mae, 'test_metrics': test_metrics
    }

def evaluate_model(model, dataloader, criterion, device, is_eval=False):
    model.eval()
    running_loss, targets, predictions = 0.0, [], []
    desc = "Evaluating" if is_eval else "Testing"
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=desc):
            inputs, labels = inputs.to(device), labels.to(device).view(-1, 1)
            inputs = inputs.unsqueeze(1).unsqueeze(2)
            inputs_rf = inputs.view(inputs.size(0), -1).cpu().numpy()
            outputs = model(inputs, inputs_rf)
            running_loss += criterion(outputs, labels).item() * inputs.size(0)
            targets.extend(labels.cpu().numpy().flatten().tolist())
            predictions.extend(outputs.cpu().numpy().flatten().tolist())
    
    avg_loss = running_loss / len(dataloader.dataset)
    # Check if targets and predictions are empty or have only one element
    if len(targets) < 2 or len(predictions) < 2:
        return {'loss': avg_loss, 'r2': 0, 'pcc': 0, 'mse': avg_loss, 'mae': np.sqrt(avg_loss)}
        
    r2 = r2_score(targets, predictions)
    pcc, _ = spearmanr(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    return {'loss': avg_loss, 'r2': r2, 'pcc': pcc, 'mse': mse, 'mae': mae}

def plot_training_history(history, test_metrics, best_epoch, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(20, 15))
    metrics_map = {
        'Loss': ('train_loss', 'val_loss', 'loss'),
        'R²': ('train_r2', 'val_r2', 'r2'),
        'PCC': ('train_pcc', 'val_pcc', 'pcc'),
        'MSE': ('train_mse', 'val_mse', 'mse'),
        'MAE': ('train_mae', 'val_mae', 'mae')
    }

    for i, (name, (train_key, val_key, test_key)) in enumerate(metrics_map.items(), 1):
        plt.subplot(3, 2, i)
        plt.plot(epochs, history[train_key], 'b-', label=f'Training {name}')
        plt.plot(epochs, history[val_key], 'r-', label=f'Validation {name}')
        plt.axhline(y=test_metrics[test_key], color='g', linestyle='-', label=f'Test {name}: {test_metrics[test_key]:.4f}')
        plt.axvline(x=best_epoch, color='k', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
        plt.title(f'Training and Validation {name}'); plt.xlabel('Epochs'); plt.ylabel(name); plt.legend(); plt.grid(True)

    if 'targets' in test_metrics and 'predictions' in test_metrics:
        plt.subplot(3, 2, 6)
        plt.scatter(test_metrics['targets'], test_metrics['predictions'], alpha=0.5)
        plt.plot([min(test_metrics['targets']), max(test_metrics['targets'])],
                [min(test_metrics['targets']), max(test_metrics['targets'])], 'r--')
        plt.title(f'Test Predictions vs Truth\nPCC={test_metrics["pcc"]:.4f}, R²={test_metrics["r2"]:.4f}')
        plt.xlabel('Ground Truth'); plt.ylabel('Predictions'); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Training history plot saved to {save_path}")

def analyze_feature_importance(model, dataloader, feature_names, device, top_n, plot_save_path, csv_save_path):
    model.eval()
    feature_importance = np.zeros(len(feature_names))
    criterion = nn.MSELoss()
    
    logging.info("\nStarting feature importance calculation...")
    for inputs, targets in tqdm(dataloader, desc="Calculating feature importance"):
        inputs, targets = inputs.to(device), targets.to(device).view(-1, 1)
        inputs = inputs.unsqueeze(1).unsqueeze(2)
        inputs.requires_grad = True
        x_rf = inputs.view(inputs.size(0), -1).detach().cpu().numpy()
        
        outputs = model(inputs, x_rf)
        loss = criterion(outputs, targets)
        model.zero_grad()
        loss.backward()
        
        grads = inputs.grad.detach().cpu().numpy()
        grads = np.abs(grads).reshape(grads.shape[0], -1)
        feature_importance += np.sum(grads, axis=0)
    
    feature_importance /= len(dataloader.dataset)
    
    top_indices = np.argsort(feature_importance)[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = feature_importance[top_indices]
    
    logging.info(f"\nTop {top_n} most important features:")
    for i, (feature, importance) in enumerate(zip(top_features, top_importance)):
        logging.info(f"{i+1}. {feature}: {importance:.6f}")
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(top_features)), top_importance, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance Score'); plt.title(f'Top {top_n} Feature Importance'); plt.tight_layout()
    plt.savefig(plot_save_path)
    plt.close()
    logging.info(f"Feature importance plot saved to {plot_save_path}")
    
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    importance_df.to_csv(csv_save_path, index=False)
    logging.info(f"Feature importance data saved to {csv_save_path}")

if __name__ == "__main__":
    try:
        pheno_df = pd.read_csv(PHENO_DATA_PATH, sep='\t')
        target_pheno_col_name = pheno_df.columns[PHENO_TARGET_COL_IDX]
    except Exception as e:
        print(f"Error: Unable to read main phenotype file - {e}."); exit()

    base_output_dir = OUTPUT_DIR if OUTPUT_DIR else os.path.dirname(PHENO_DATA_PATH)
    if not base_output_dir: base_output_dir = '.'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    training_folder_name = f"{target_pheno_col_name}_{timestamp}"
    output_dir = os.path.join(base_output_dir, training_folder_name)
    os.makedirs(output_dir, exist_ok=True)
    
    log_path = os.path.join(output_dir, f"{target_pheno_col_name}_{timestamp}.log")
    setup_logging(log_path)
    save_hyperparameters(output_dir, target_pheno_col_name, timestamp)
    
    logging.info("========================= GenoFuse Model Training Started =========================")
    torch.manual_seed(SEED); np.random.seed(SEED)
    logging.info(f"Using device: {DEVICE}")
    
    logging.info("Loading main dataset...")
    geno_df = pd.read_csv(GENO_DATA_PATH, sep=r'\s+', low_memory=True)
    pheno_to_merge = pheno_df[[PHENO_ID_COL, target_pheno_col_name]]
    main_data = pd.merge(geno_df, pheno_to_merge, left_on=GENO_ID_COL, right_on=PHENO_ID_COL, how='inner')
    
    if main_data.empty: logging.error("Error: No overlapping IDs in main dataset."); exit()
    
    main_data[main_data.columns[5]] = main_data[target_pheno_col_name]
    main_data = main_data.drop(columns=[PHENO_ID_COL, target_pheno_col_name])
    main_data = handle_outliers(main_data, list(main_data.columns[6:]), threshold=OUTLIER_THRESHOLD)
    main_input, main_target = main_data.iloc[:, 6:].values, main_data.iloc[:, 5].values
    feature_names = list(main_data.columns[6:])
    
    scaler = RobustScaler(); scaled_main_input = scaler.fit_transform(main_input)
    full_dataset = TensorDataset(torch.tensor(scaled_main_input, dtype=torch.float32), 
                                 torch.tensor(main_target, dtype=torch.float32))

    if USE_CROSS_VALIDATION:
        logging.info(f"Enabling {N_SPLITS}-fold cross-validation...")
        kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        fold_results = []
            
        independent_test_dataloader = None
        if TEST_GENO_DATA_PATH and TEST_PHENO_DATA_PATH:
            logging.info("Loading independent test set for cross-validation evaluation...")
            test_geno_df = pd.read_csv(TEST_GENO_DATA_PATH, sep=r'\s+', low_memory=False)
            test_pheno_df = pd.read_csv(TEST_PHENO_DATA_PATH, sep='\t')
            test_pheno_to_merge = test_pheno_df[[PHENO_ID_COL, target_pheno_col_name]]
            test_data = pd.merge(test_geno_df, test_pheno_to_merge, on=GENO_ID_COL, how='inner')
            test_data[test_data.columns[5]] = test_data[target_pheno_col_name]
            test_data = test_data.drop(columns=[PHENO_ID_COL, target_pheno_col_name])
            test_input = scaler.transform(test_data.iloc[:, 6:].values)
            test_target = test_data.iloc[:, 5].values
            test_dataset = TensorDataset(torch.tensor(test_input, dtype=torch.float32), 
                                         torch.tensor(test_target, dtype=torch.float32))
            independent_test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)

        for fold, (train_ids, val_ids) in enumerate(kfold.split(full_dataset)):
            fold_output_dir = os.path.join(output_dir, f'fold_{fold + 1}')
            os.makedirs(fold_output_dir, exist_ok=True)
            logging.info(f"===================== Starting Fold {fold + 1}/{N_SPLITS} (Results saved to: {fold_output_dir}) =====================")
            
            train_dataloader = DataLoader(Subset(full_dataset, train_ids), batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
            val_dataloader = DataLoader(Subset(full_dataset, val_ids), batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
            test_dataloader_for_fold = independent_test_dataloader if independent_test_dataloader else val_dataloader
            
            logging.info(f"[Fold {fold+1}] Training Random Forest...")
            rf_model = RandomForestRegressor(**RF_PARAMS).fit(scaled_main_input[train_ids], main_target[train_ids])

            deep_model = Conformer(**CONFORMER_PARAMS).to(DEVICE)
            if USE_MULTI_GPU and len(GPU_IDS) > 1:
                deep_model = nn.DataParallel(deep_model, device_ids=GPU_IDS)
            
            optimizer = optim.AdamW(deep_model.parameters(), **OPTIMIZER_PARAMS)
            scheduler = CosineAnnealingWarmRestarts(optimizer, **SCHEDULER_PARAMS)
            
            es_params = EARLY_STOPPING_PARAMS.copy()
            es_params['save_path'] = os.path.join(fold_output_dir, "best_model.pth")
            es_params['save_model'] = SAVE_PER_FOLD_MODELS

            _, history = train_model(
                model=deep_model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
                test_dataloader=test_dataloader_for_fold, criterion=CRITERION, optimizer=optimizer,
                scheduler=scheduler, device=DEVICE, num_epochs=NUM_EPOCHS, rf_model=rf_model,
                dynamic_rf_weight=DYNAMIC_RF_WEIGHT, early_stopping_params=es_params,
                history_plot_path=os.path.join(fold_output_dir, "training_history.png"),
                accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                save_history_plot=SAVE_PER_FOLD_PLOTS
            )
            fold_results.append(history['test_metrics'])

        logging.info(f"\n===================== {N_SPLITS}-Fold Cross-Validation Summary =====================")
        all_pcc = [res['pcc'] for res in fold_results]; all_r2 = [res['r2'] for res in fold_results]
        all_mse = [res['mse'] for res in fold_results]; all_mae = [res['mae'] for res in fold_results]

        logging.info(f"Average PCC: {np.mean(all_pcc):.4f} ± {np.std(all_pcc):.4f}")
        logging.info(f"Average R²:  {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
        logging.info(f"Average MSE: {np.mean(all_mse):.4f} ± {np.std(all_mse):.4f}")
        logging.info(f"Average MAE: {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
    
        logging.info("Standard training workflow initiated (no cross-validation).")
        
        train_dataset, val_dataset, test_dataset = None, None, None
        
        if TEST_GENO_DATA_PATH and TEST_PHENO_DATA_PATH:
            logging.info("Using independent test set, splitting main data into train/validation sets...")
            train_size = int(TRAIN_VALIDATION_SPLIT_RATIO * len(full_dataset))
            val_size = len(full_dataset) - train_size
            train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED))
            
            # Load independent test set
            test_geno_df = pd.read_csv(TEST_GENO_DATA_PATH, sep=r'\s+', low_memory=False)
            test_pheno_df = pd.read_csv(TEST_PHENO_DATA_PATH, sep='\t')
            test_pheno_to_merge = test_pheno_df[[PHENO_ID_COL, target_pheno_col_name]]
            test_data = pd.merge(test_geno_df, test_pheno_to_merge, on=GENO_ID_COL, how='inner')
            test_data[test_data.columns[5]] = test_data[target_pheno_col_name]
            test_data = test_data.drop(columns=[PHENO_ID_COL, target_pheno_col_name])
            test_input = scaler.transform(test_data.iloc[:, 6:].values)
            test_target = test_data.iloc[:, 5].values
            test_dataset = TensorDataset(torch.tensor(test_input, dtype=torch.float32), 
                                         torch.tensor(test_target, dtype=torch.float32))
        else:
            logging.info("No independent test set, splitting main data into train/validation/test sets...")
            train_size = int(TRAIN_SPLIT_RATIO * len(full_dataset))
            val_size = int(VALIDATION_SPLIT_RATIO * len(full_dataset))
            test_size = len(full_dataset) - train_size - val_size
            train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(SEED))
        
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
        val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
        test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=PIN_MEMORY, num_workers=NUM_WORKERS)
        
        logging.info("Training Random Forest model...")
        train_indices = train_dataset.indices if isinstance(train_dataset, Subset) else list(range(len(train_dataset)))
        rf_model = RandomForestRegressor(**RF_PARAMS).fit(scaled_main_input[train_indices], main_target[train_indices])
        
        deep_model = Conformer(**CONFORMER_PARAMS).to(DEVICE)
        if USE_MULTI_GPU and len(GPU_IDS) > 1:
            deep_model = nn.DataParallel(deep_model, device_ids=GPU_IDS)
        
        optimizer = optim.AdamW(deep_model.parameters(), **OPTIMIZER_PARAMS)
        scheduler = CosineAnnealingWarmRestarts(optimizer, **SCHEDULER_PARAMS)
        
        es_params = EARLY_STOPPING_PARAMS.copy()
        es_params['save_path'] = os.path.join(output_dir, f"{target_pheno_col_name}_{timestamp}_best_model.pth")
        es_params['save_model'] = True # Always save model in standard mode

        ensemble_model, history = train_model(
            model=deep_model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,
            test_dataloader=test_dataloader, criterion=CRITERION, optimizer=optimizer,
            scheduler=scheduler, device=DEVICE, num_epochs=NUM_EPOCHS, rf_model=rf_model,
            dynamic_rf_weight=DYNAMIC_RF_WEIGHT, early_stopping_params=es_params,
            history_plot_path=os.path.join(output_dir, f"{target_pheno_col_name}_{timestamp}_training_history.png"),
            accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            save_history_plot=True # Always save plots in standard mode
        )

        analyze_feature_importance(
            ensemble_model, val_dataloader, feature_names, DEVICE,
            top_n=FEATURE_IMPORTANCE_TOP_N,
            plot_save_path=os.path.join(output_dir, f"{target_pheno_col_name}_{timestamp}_feature_importance.png"),
            csv_save_path=os.path.join(output_dir, f"{target_pheno_col_name}_{timestamp}_feature_importance.csv")
        )
    
    logging.info("========================= GenoFuse Model Training Completed =========================")