# OmniTFT: Multi-target forecasting for vital signs and laboratory trajectories in critically ill patients: an interpretable deep learning model

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10.0-orange.svg)](https://www.tensorflow.org/)
[![CUDA 11.3](https://img.shields.io/badge/CUDA-11.3-green.svg)](https://developer.nvidia.com/cuda-toolkit)

An enhanced Temporal Fusion Transformer (TFT) implementation for unified forecasting of vital signs and laboratory trajectories in ICU medical data, building upon the original TFT architecture by [Lim et al. (2021)](https://doi.org/10.1016/j.ijforecast.2021.03.012) with four novel strategies to improve prediction accuracy and clinical interpretability.

## Abstract

Accurate multivariate time-series prediction of vital signs and laboratory results is crucial for early intervention and precision medicine in intensive care units (ICUs). However, vital signs are often noisy and exhibit rapid fluctuations, while laboratory tests suffer from missing values, measurement lags, and device-specific bias, making integrative forecasting highly challenging.

To address these issues, we propose **OmniTFT**, a deep learning framework that jointly learns and forecasts high-frequency vital signs and sparsely sampled laboratory results based on the Temporal Fusion Transformer (TFT). Specifically, OmniTFT implements four novel strategies to enhance performance:

1. **Sliding window equalized sampling** to balance physiological states
2. **Frequency-aware embedding shrinkage** to stabilize rare-class representations
3. **Hierarchical variable selection** to guide model attention toward informative feature clusters
4. **Influence-aligned attention calibration** to enhance robustness during abrupt physiological changes

By reducing the reliance on target-specific architectures and extensive feature engineering, OmniTFT enables unified modeling of multiple heterogeneous clinical targets while preserving cross-institutional generalizability. Across forecasting tasks, OmniTFT achieves substantial performance improvement for both vital signs and laboratory results on the **MIMIC-III, MIMIC-IV, and eICU datasets**. Its attention patterns are interpretable and consistent with known pathophysiology, underscoring its potential utility for quantitative decision support in clinical care.

## Overview

OmniTFT is designed for multivariate time-series forecasting in intensive care unit (ICU) settings. The model predicts critical physiological parameters using historical patient data, providing probabilistic forecasts with quantile predictions (10th, 50th, and 90th percentiles).

### Key Features

📊 **Supported Clinical Targets:**
- Respiratory Rate
- Lactate
- Creatinine
- Heart Rate
- Blood Pressure
- Oxygen Saturation
- Temperature
- SF Ratio
- *Extensible to other ICU parameters*

## Installation

### Prerequisites

- **CUDA 11.3** and **cuDNN 8.2** (for GPU support)
- **Anaconda** or **Miniconda**
- **Windows/Linux** operating system

### Environment Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/OmniTFT.git
   cd OmniTFT
   ```

2. **Create conda environment:**
   ```bash
   conda env create -f icu_env.yml
   conda activate tfticu
   ```
   The environment includes:
   - Python 3.8.20
   - TensorFlow 2.10.0 (GPU)
   - CUDA Toolkit 11.3.1
   - cuDNN 8.2.1
   - scikit-learn, pandas, numpy
   - matplotlib, seaborn
   - SHAP for interpretability

## Usage

### Training a Single Target

Train a model for a specific clinical target (e.g., Respiratory Rate):

```bash
python Train_all_target.py <output_folder> <use_gpu>

# Example (using defaults):
python Train_all_target.py . yes
```

**Arguments:**
- `output_folder`: Directory to save model outputs and results (default: `.`)
- `use_gpu`: Whether to use GPU for training (`yes` or `no`, default: `yes`)

### Training Multiple Targets

The training script automatically trains models for all targets listed in `FORMATTERS_TO_TRAIN`:
Here we only show three files as an example. If you wanna to add new labels you need to add new files here.
```python
FORMATTERS_TO_TRAIN = [
    "Lactate",
    "RespiratoryRate",
    "Creatinine",
    "new_target"
]
```

### Output Structure

After training, the following structure will be created:

```
outputs/
├── data/<target>/              # CSV data files
├── saved_models/<target>/      # Model checkpoints and configurations
│   └── fixed/
│       ├── tmp/                # Temporary Keras checkpoints
│       ├── scalers/            # JSON files with scaler parameters
│       ├── OmniTFT.ckpt        # Best model weights
│       ├── model_config.json   # Model configuration
│       └── best_result.json    # Best hyperparameters and loss
└── results/<target>/           # Prediction results
```

## Project Structure

```
OmniTFT/
├── libs/                           # Core model components
│   ├── OmniTFT.py                 # Main model class
│   ├── omnitft_components.py      # Layers and attention mechanisms
│   ├── hyperparam_opt.py          # Hyperparameter optimization
│   └── utils.py                   # Utility functions
├── data_formatters/               # Data preprocessing modules
│   ├── base.py                    # Base formatter class
│   ├── RespiratoryRate.py         # Respiratory rate formatter
│   ├── Lactate.py                 # Lactate formatter
│   └── Creatinine.py              # Creatinine formatter
├── expt_settings/                 # Experiment configurations
│   └── configs.py                 # Experiment config manager
├── Train_all_target.py            # Main training script
├── icu_env.yml                    # Conda environment specification
└── README.md                      # This file
```


## Adding a New Target

1. **Create formatter**: `data_formatters/NewTarget.py`
   ```python
   class NewTargetFormatter(GenericDataFormatter):
       # Define column mappings
       _column_definition = [...]

       # Implement required methods
       def split_data(self, df): ...
       def get_fixed_params(self): ...
   ```

2. **Register in configs**: `expt_settings/configs.py`
   ```python
   csv_map = {
       'NewTarget': 'path/to/data.csv',
   }

   data_formatter_class = {
       'NewTarget': 'data_formatters.NewTarget.NewTargetFormatter',
   }
   ```

3. **Add to training list**: `Train_all_target.py`
   ```python
   FORMATTERS_TO_TRAIN = [
       "NewTarget",
       # ... existing targets
   ]
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
