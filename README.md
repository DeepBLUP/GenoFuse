# GenoFuse: Deep Learning Framework for Genomic Prediction

GenoFuse is an advanced deep learning framework that combines Conformer neural networks with Random Forest ensemble methods for genomic prediction tasks. It's specifically designed for quantitative genetics research, particularly in livestock breeding programs.

## 🚀 Features

- **Hybrid Architecture**: Combines deep learning (Conformer) with traditional machine learning (Random Forest)
- **Cross-Validation Support**: Built-in k-fold cross-validation for robust model evaluation
- **Multi-GPU Training**: Supports both DataParallel and DistributedDataParallel strategies
- **Mixed Precision Training**: FP16 support for memory efficiency and faster training
- **Feature Importance Analysis**: Comprehensive analysis of genomic marker importance
- **Early Stopping**: Prevents overfitting with configurable early stopping mechanisms
- **Flexible Data Input**: Supports standard PLINK .raw format and custom phenotype files

## 📋 Requirements

### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended (8GB+ VRAM for large datasets)
- **RAM**: Minimum 8GB, 16GB+ recommended for large genomic datasets
- **Storage**: Sufficient space for genomic data, models, and output files

### Software Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU acceleration)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/GenoFuse.git
cd GenoFuse
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify CUDA installation (optional but recommended):
```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

## 📊 Data Format

### Genotype Data (.raw format)
- Standard PLINK .raw format (space-separated)
- Must contain 'IID' column for individual identification
- SNP columns should be coded as 0, 1, 2 (additive coding)

### Phenotype Data (.txt format)
- Tab-separated format
- Must contain ID column matching genotype IID
- Can contain multiple phenotype columns

Example:
```
ID    Trait1    Trait2    Trait3
Ind1  100.5     85.2      12.3
Ind2  95.8      88.1      11.9
```

## ⚙️ Configuration

The model provides extensive configuration options divided into three categories:

### 1. File and Path Configuration
```python
GENO_DATA_PATH = '/path/to/genotype.raw'     # [REQUIRED] Genotype file path
PHENO_DATA_PATH = '/path/to/phenotype.txt'   # [REQUIRED] Phenotype file path
TEST_GENO_DATA_PATH = None                   # [OPTIONAL] Independent test genotype
TEST_PHENO_DATA_PATH = None                  # [OPTIONAL] Independent test phenotype
OUTPUT_DIR = None                            # [OPTIONAL] Output directory
```

### 2. Basic Parameter Tuning (Commonly Adjusted)

#### Training Parameters
- `SEED = 123`: Random seed for reproducibility
- `BATCH_SIZE = 30`: Training batch size (adjust based on GPU memory)
- `NUM_EPOCHS = 120`: Number of training epochs
- `GRADIENT_ACCUMULATION_STEPS = 6`: Gradient accumulation for effective larger batch sizes

#### Learning Rate and Optimization
- `LEARNING_RATE = 0.002`: Initial learning rate
- `WEIGHT_DECAY = 0.003`: L2 regularization strength

#### Regularization (Dropout Rates)
- `DROP_RATE = 0.4`: General dropout rate
- `ATTN_DROP_RATE = 0.3`: Attention mechanism dropout
- `DROP_PATH_RATE = 0.2`: Stochastic depth dropout

#### Cross-Validation
- `USE_CROSS_VALIDATION = True`: Enable k-fold cross-validation
- `N_SPLITS = 5`: Number of folds
- `SAVE_PER_FOLD_MODELS = True`: Save individual fold models
- `SAVE_PER_FOLD_PLOTS = True`: Save training plots per fold

#### Early Stopping
- `EARLY_STOPPING_PATIENCE = 12`: Epochs to wait before stopping
- `EARLY_STOPPING_MIN_DELTA = 0.008`: Minimum improvement threshold

### 3. Advanced Parameter Tuning (Architecture & Technical)

#### Model Architecture
- `PATCH_SIZE = 4`: Input patch size for Conformer
- `BASE_CHANNEL = 16`: Base number of channels
- `EMBED_DIM = 24`: Embedding dimension
- `DEPTH = 3`: Number of transformer layers
- `NUM_HEADS = 2`: Multi-head attention heads

#### Random Forest Ensemble
- `RF_N_ESTIMATORS = 50`: Number of trees
- `RF_MAX_DEPTH = 8`: Maximum tree depth
- `RF_MAX_FEATURES = 'sqrt'`: Features per split
- `RF_MIN_SAMPLES_SPLIT = 10`: Minimum samples to split
- `RF_MIN_SAMPLES_LEAF = 5`: Minimum samples per leaf

#### Memory Optimization
- `USE_MIXED_PRECISION = True`: FP16 training
- `USE_GRADIENT_CHECKPOINTING = True`: Memory-efficient backprop
- `PIN_MEMORY = False`: Pinned memory for data loading
- `NUM_WORKERS = 0`: Data loader workers

#### Multi-GPU Configuration
- `USE_MULTI_GPU = True`: Enable multi-GPU training
- `MULTI_GPU_STRATEGY = "DataParallel"`: Parallelization strategy
- `GPU_IDS = [0, 1, 2]`: GPU devices to use

## 🚀 Usage

### Basic Usage
1. Configure your data paths in the script:
```python
GENO_DATA_PATH = '/path/to/your/genotype.raw'
PHENO_DATA_PATH = '/path/to/your/phenotype.txt'
PHENO_TARGET_COL_IDX = 1  # Which phenotype column to predict (0-indexed)
```

2. Run the training:
```bash
python GenoFuse_5fold_CV.py
```

### Advanced Usage

#### Cross-Validation Mode (Default)
```python
USE_CROSS_VALIDATION = True
N_SPLITS = 5
```
Performs k-fold cross-validation and reports average performance metrics.

#### Independent Test Set Mode
```python
USE_CROSS_VALIDATION = False
TEST_GENO_DATA_PATH = '/path/to/test_genotype.raw'
TEST_PHENO_DATA_PATH = '/path/to/test_phenotype.txt'
```
Uses independent test set for final evaluation.

#### Single GPU Training
```python
USE_MULTI_GPU = False
BATCH_SIZE = 4  # Smaller batch size for single GPU
```

## 📈 Output Files

The framework generates several output files:

### Model Files
- `{phenotype}_{timestamp}_best_model.pth`: Best model weights
- `{phenotype}_{timestamp}_fold_{i}_model.pth`: Per-fold models (if enabled)

### Visualization
- `{phenotype}_{timestamp}_training_history.png`: Training/validation curves
- `{phenotype}_{timestamp}_feature_importance.png`: Feature importance plot

### Analysis Results
- `{phenotype}_{timestamp}_feature_importance.csv`: Feature importance scores
- `{phenotype}_{timestamp}_training.log`: Detailed training logs

### Performance Metrics
- R² (coefficient of determination)
- RMSE (root mean squared error)
- MAE (mean absolute error)
- Spearman correlation coefficient

## 🔧 Hyperparameter Tuning Guide

### For Small Datasets (<1000 samples)
```python
BATCH_SIZE = 8
LEARNING_RATE = 0.001
DEPTH = 2
NUM_HEADS = 1
RF_N_ESTIMATORS = 30
```

### For Large Datasets (>10000 samples)
```python
BATCH_SIZE = 64
LEARNING_RATE = 0.003
DEPTH = 4
NUM_HEADS = 4
RF_N_ESTIMATORS = 100
```

### For Limited GPU Memory
```python
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
USE_MIXED_PRECISION = True
USE_GRADIENT_CHECKPOINTING = True
PIN_MEMORY = False
NUM_WORKERS = 0
```

## 🧬 Model Architecture

GenoFuse employs a hybrid architecture:

1. **Conformer Network**: Combines convolutional and transformer layers for genomic sequence modeling
2. **Random Forest**: Captures non-linear interactions and provides ensemble diversity
3. **Dynamic Weighting**: Adaptively combines predictions from both models

### Key Components
- **Attention Mechanism**: Multi-head self-attention for long-range dependencies
- **Convolutional Blocks**: Local feature extraction from genomic markers
- **Feature Fusion**: Combines CNN and transformer representations
- **Ensemble Integration**: Weighted combination of deep and tree-based predictions

## 📚 Citation

If you use GenoFuse in your research, please cite:

```bibtex
@software{genofuse2024,
  title={GenoFuse: Deep Learning Framework for Genomic Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/GenoFuse}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE`
   - Enable `USE_MIXED_PRECISION`
   - Set `USE_GRADIENT_CHECKPOINTING = True`

2. **Slow Training**
   - Increase `BATCH_SIZE` if memory allows
   - Use multiple GPUs with `USE_MULTI_GPU = True`
   - Reduce `NUM_WORKERS` if CPU is bottleneck

3. **Poor Performance**
   - Increase model complexity (`DEPTH`, `EMBED_DIM`)
   - Adjust learning rate and regularization
   - Check data quality and preprocessing

4. **Memory Issues**
   - Set `PIN_MEMORY = False`
   - Reduce `NUM_WORKERS` to 0
   - Use smaller batch sizes

## 📞 Support

For questions and support, please:
1. Check the troubleshooting section above
2. Open an issue on GitHub
3. Contact the development team

---

**Note**: This framework is specifically designed for quantitative genetics research and genomic prediction tasks. It has been tested on livestock breeding datasets but can be adapted for other genomic prediction applications.