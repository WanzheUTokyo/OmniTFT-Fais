# OmniTFT: Temporal Fusion Transformer for ICU Clinical Time-Series Forecasting

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

OmniTFT is an advanced deep learning framework for multi-horizon clinical time-series forecasting in Intensive Care Units (ICU). Built on the Temporal Fusion Transformer architecture, it incorporates three novel regularization techniques designed specifically for critical care applications.


## Project Structure

```
OmniTFT/
├── components/              # Neural network building blocks
│   ├── attention_layers.py     # Multi-head attention mechanisms
│   ├── embedding_layers.py     # Regularization layers & utilities
│   └── __init__.py
├── core/                    # Core model architecture
│   ├── omnitft_model.py        # Main OmniTFT model class
│   └── __init__.py
├── training/                # Training utilities
│   ├── hyperparam_optimizer.py # Hyperparameter optimization
│   ├── training_utils.py       # Training helper functions
│   └── __init__.py
├── data_formatter/          # Targets definition
│   ├── base_formatter.py       # Abstract base class
│   ├── lactate_formatter.py    # Lactate prediction formatter
│   ├── creatinine_formatter.py # Creatinine prediction formatter
│   ├── respiratory_formatter.py# Respiratory rate formatter
│   └── __init__.py
├── config/                  # Experiment configuration
│   ├── experiment_setup.py     # Experiment settings manager
│   └── __init__.py
├── train_pipeline.py        # Main training script
├── icu_env.yml             # Conda environment specification
└── README.md
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.3+ (for GPU support)
- cuDNN 8.2+

### Environment Setup

Using Conda (recommended):

```bash
# Create environment from yml file
conda env create -f icu_env.yml

# Activate environment
conda activate icu_env
```


### Key Dependencies

- **TensorFlow**: 2.10.0 (GPU)
- **NumPy**: 1.24.4
- **Pandas**: 2.0.3
- **Scikit-learn**: 1.3.2
- **SHAP**: 0.42.1 (for model interpretability)

## Quick Start

### Basic Usage

```bash
# Train on all targets with GPU
python train_pipeline.py . yes

# Train on all targets with CPU
python train_pipeline.py . no

# Specify custom output folder
python train_pipeline.py /path/to/output yes
```

### Supported Clinical Targets

- **Lactate**: Serum lactate level prediction
- **Creatinine**: Serum creatinine level prediction
- **Respiratory Rate**: Respiratory rate forecasting
- **Could add more**

### Training Configuration

Edit `FORMATTERS_TO_TRAIN` in `train_pipeline.py` to select targets:

```python
FORMATTERS_TO_TRAIN = [
    "Lactate",
    "RespiratoryRate",
    "Creatinine",
    "Others"
]
```

## Data Format

### Input Requirements

Data should be in CSV format with the example data format in output -> data folder:

```

### Output Structure

```
outputs/
├── data/
│   └── <target>/
│       ├── train.csv
│       ├── valid.csv
│       └── test.csv
├── saved_models/
│   └── <target>/
│       └── fixed/
│           ├── OmniTFT.ckpt
│           ├── model_config.json
│           └── scalers/
│               └── all_scalers.json
└── results/
    └── <target>/
        ├── p10_forecast.csv
        ├── p50_forecast.csv
        └── p90_forecast.csv
```

## Advanced Usage

### Custom Data Formatter

Create a new formatter by inheriting from `GenericDataFormatter`:

### GPU Utilization

```python
# Enable GPU memory growth
tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True

# Set visible GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```
## Citation

If you use OmniTFT in your research, please cite:

```
```

## License

This project is licensed under the MIT License

## Acknowledgments
Computational resources were provided by the supercomputer system SHIROKANE at the Human Genome Center,  Institute of Medical Science, University of Tokyo.
This work was supported by JST SPRING, grant number JPMJSP2108.

## Contact

For questions or issues, please:
- Contact: xu-wanzhe884@g.ecc.u-tokyo.ac.jp

## Changelog

### Version 1.0.0 (2025)

---

**Note**: This is a research implementation. For clinical deployment, please ensure regulatory compliance.
