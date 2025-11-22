
# Lint as: python3
"""Trains OmniTFT based on a defined set of parameters.

Usage:
python3 OmniTFT_training.py {output_folder} --is_GPU
"""

import matplotlib
import argparse
import datetime as dte
import os
import pandas as pd
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import sklearn.preprocessing
import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.OmniTFT
import libs.utils as utils
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
tf_config = tf.ConfigProto(
    inter_op_parallelism_threads=1,
    intra_op_parallelism_threads=1,
    allow_soft_placement=True,
    log_device_placement=True
)

from tensorflow.python.client import device_lib
print("=== Local devices ===")
for d in device_lib.list_local_devices():
    print(d)


from data_formatters.RespiratoryRate import RespiratoryRateFormatter, DataTypes, InputTypes
from data_formatters.Creatinine import CreatinineFormatter, DataTypes, InputTypes
from data_formatters.Lactate import LactateFormatter, DataTypes, InputTypes

import matplotlib.pyplot as plt
ExperimentConfig = expt_settings.configs.ExperimentConfig
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager
ModelClass = libs.OmniTFT.OmniTFT
tf.experimental.output_all_intermediates(True)
    # "Temperature",
    # "BP",
    # "Bun",
    # "HR",
    # "OxygenSaturation",
    # "OxygenationIndex",
FORMATTERS_TO_TRAIN = [
    "Lactate",
    "RespiratoryRate",
    "Creatinine",
]

def main(expt_name,
         use_gpu,
         model_folder,
         data_csv_path,
         data_formatter,
         use_testing_mode=False):

    val_losses, p10_losses, p50_losses, p90_losses = [], [], [], []
    global global_val_loss, global_p10_loss, global_p50_loss, global_p90_loss

    num_repeats = 1

    if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
        raise ValueError("Data formatters should inherit from GenericDataFormatter!")

    default_keras_session = tf.keras.backend.get_session()
    tf_config = utils.get_default_tensorflow_config(
        tf_device="gpu" if use_gpu else "cpu", gpu_id=0)

    print("*** Training from defined parameters for {} ***".format(expt_name))

    print("Loading & splitting data...")
    raw_data = pd.read_csv(data_csv_path, index_col=0)
    train, valid, test = data_formatter.split_data(raw_data)
    train_samples, valid_samples = data_formatter.get_num_samples_for_calibration()

    scaler_folder = os.path.join(model_folder, "scalers")
    os.makedirs(scaler_folder, exist_ok=True)
    data_formatter.save_scalers(scaler_folder)


    fixed_params = data_formatter.get_experiment_params()
    params = data_formatter.get_default_model_params()
    params["model_folder"] = model_folder
    params = {**params, **fixed_params, "model_folder": model_folder}

    if use_testing_mode:
        fixed_params["num_epochs"] = 1
        params["hidden_layer_size"] = 5
        train_samples, valid_samples = 100, 10

    opt_manager = HyperparamOptManager({k: [params[k]] for k in params},
                                       fixed_params, model_folder)

    best_loss = np.Inf
    for _ in range(num_repeats):

        tf.reset_default_graph()
        with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
            tf.keras.backend.set_session(sess)

            model = ModelClass(params, use_cudnn=use_gpu)

            if not model.training_data_cached():
                model.cache_batched_data(train, "train", num_samples=train_samples)
                model.cache_batched_data(valid, "valid", num_samples=valid_samples)

            sess.run(tf.global_variables_initializer())
            history = model.fit()

            val_loss = model.evaluate()
            val_losses.append(val_loss)

            if val_loss < best_loss:
                opt_manager.update_score(params, val_loss, model)
                best_loss = val_loss

            tf.keras.backend.set_session(default_keras_session)

    print("*** Running tests ***")
    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
        tf.keras.backend.set_session(sess)
        best_params = opt_manager.get_best_params()
        model = ModelClass(best_params, use_cudnn=use_gpu)
        model.load(opt_manager.hyperparam_folder)

        val_loss = model.evaluate(valid)
        output_map = model.predict(test, return_targets=True)
        targets = data_formatter.format_predictions(output_map["targets"])
        p10_forecast = data_formatter.format_predictions(output_map["p10"])
        p50_forecast = data_formatter.format_predictions(output_map["p50"])
        p90_forecast = data_formatter.format_predictions(output_map["p90"])

        def strip(df):
            return df[[c for c in df.columns if c not in {"forecast_time", "identifier"}]]

        p10_loss = utils.numpy_normalised_quantile_loss(strip(targets), strip(p10_forecast), 0.1)
        p50_loss = utils.numpy_normalised_quantile_loss(strip(targets), strip(p50_forecast), 0.5)
        p90_loss = utils.numpy_normalised_quantile_loss(strip(targets), strip(p90_forecast), 0.9)
        tf.keras.backend.set_session(default_keras_session)

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(val_loss))
    for k, v in best_params.items():
        print(k, "=", v)
    p10_losses.append(p10_loss.mean()); p50_losses.append(p50_loss.mean()); p90_losses.append(p90_loss)
    global_val_loss, global_p10_loss, global_p50_loss, global_p90_loss = val_losses, p10_losses, p50_losses, p90_losses
    print("Normalised Quantile Loss for Test Data: P10={}, P50={}, P90={}".format(
        p10_loss.mean(), p50_loss.mean(), p90_loss.mean()))


if __name__ == "__main__":
    def get_args():
        parser = argparse.ArgumentParser(description="Train TFT models for all targets")
        parser.add_argument("output_folder", metavar="output_folder", type=str, nargs="?", default=".",
                            help="Output folder path (default: current directory)")
        parser.add_argument("use_gpu", metavar="use_gpu", type=str, nargs="?", choices=["yes", "no"], default="yes",
                            help="Use GPU for training (yes/no, default: yes)")
        args = parser.parse_known_args()[0]
        root_folder = None if args.output_folder == "." else args.output_folder
        return root_folder, args.use_gpu == "yes"

    output_folder, use_tensorflow_with_gpu = get_args()
    print("Using output folder {}".format(output_folder))
    print("Training all targets: {}".format(", ".join(FORMATTERS_TO_TRAIN)))

    for name in FORMATTERS_TO_TRAIN:
        print("\n" + "="*80)
        print(f"Starting training for: {name}")
        print("="*80 + "\n")
        
        config = ExperimentConfig(name, output_folder)
        formatter = config.make_data_formatter()

        main(
            expt_name=name,
            use_gpu=use_tensorflow_with_gpu,
            model_folder=os.path.join(config.model_folder, "fixed"),
            data_csv_path=config.data_csv_path,
            data_formatter=formatter,
            use_testing_mode=False)
        
        print(f"\n✓ Completed training for: {name}\n")
        print("Cleaning up resources...")
        tf.keras.backend.clear_session()
        tf.reset_default_graph()
        libs.OmniTFT.TFTDataCache._data_cache.clear()
        import gc
        gc.collect()
        print("Resources cleaned.\n")



