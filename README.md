# OmniTFT: Omni Target Forecasting for Vital Signs and Laboratory Result Trajectories in Multi Center ICU Data

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
├── multi_task_predictor.py  # Simultaneous multi-task inference utility
├── icu_env.yml             # Conda environment specification
└── README.md
```

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

## Quick Start

### Basic Usage

```bash
# Train on all targets with GPU
python train_pipeline.py . yes

# Train on all targets with CPU
python train_pipeline.py . no

# Train all targets in parallel
python train_pipeline.py . yes --parallel

# Specify custom output folder
python train_pipeline.py /path/to/output yes

# Train a single target
python train_pipeline.py . yes --task Lactate
```

### Parallel Training and Multi-Task Inference

- By default, `train_pipeline.py` trains all configured targets sequentially.
- With `--parallel`, each target is launched in an independent process and assigned a GPU automatically when available.
- In parallel mode, each target writes to its own log file under `logs/parallel_<timestamp>/`.
- After all target-specific training jobs finish, the pipeline automatically loads all trained models and runs multi-task simultaneous inference.
- Prediction outputs are saved both per target and as merged combined files in `outputs/results/combined/`.

### Standalone Multi-Task Inference

You can also load trained checkpoints directly and run joint inference without retraining:

```python
from multi_task_predictor import MultiTaskPredictor

predictor = MultiTaskPredictor.load_from_saved(
    model_root='outputs/saved_models',
    tasks=['Creatinine', 'Lactate', 'RespiratoryRate'],
    use_gpu=True)

results = predictor.predict_all(test_data_override={
    'Creatinine': creatinine_df,
    'Lactate': lactate_df,
    'RespiratoryRate': respiratory_df,
})

combined_df = predictor.get_combined_predictions(quantile='p50')
predictor.save_results('outputs/results')
predictor.close()
```

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

Here we only show three files as an example. If you wanna to add new labels you need to add new files here.
```python
FORMATTERS_TO_TRAIN = [
    "Lactate",
    "RespiratoryRate",
    "Creatinine",
    "new_target"
]
```

## Data Format

### Input Requirements

Data should be in CSV format with the example data format in output -> data folder:

## Advanced Usage

### Custom new target

To add a new clinical target for prediction, follow these steps:

#### Step 1: Create a New Formatter File

Create `data_formatter/your_target_formatter.py` (e.g., `heartrate_formatter.py`):

#### Step 2: Register in Configuration

Edit `config/experiment_setup.py`

#### Step 3: Add to Training Pipeline

Edit `train_pipeline.py`

#### Step 4: Prepare Your Data

Place your CSV file in `outputs/data/YourTarget/your_target.csv` with the required columns matching your `_column_definition`.

#### Step 5: Run Training

```bash
python train_pipeline.py . yes
```

The framework will automatically:
- Load your data
- Apply your formatter
- Train the model
- Save checkpoints to `outputs/saved_models/YourTarget/`
- Save per-target predictions to `outputs/results/YourTarget/`
- Save merged multi-task predictions to `outputs/results/combined/`

## Citation

If you use OmniTFT in your research, please cite:

```
```
## Reference
Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). Temporal fusion transformers for interpretable multi-horizon time series forecasting. International journal of forecasting, 37(4), 1748-1764.
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
