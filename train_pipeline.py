
# Lint as: python3
"""Trains OmniTFT based on a defined set of parameters.

Usage:
python3 train_pipeline.py {output_folder} {use_gpu}
"""

import matplotlib
import argparse
import datetime as dte
import os
import subprocess
import sys
import pandas as pd
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

import sklearn.preprocessing
import data_formatter.base_formatter as base_fmt
import config.experiment_setup
import training.hyperparam_optimizer
import core.omnitft_model
import training.training_utils as utils
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


from data_formatter.respiratory_formatter import RespiratoryRateFormatter, DataTypes, InputTypes
from data_formatter.creatinine_formatter import CreatinineFormatter, DataTypes, InputTypes
from data_formatter.lactate_formatter import LactateFormatter, DataTypes, InputTypes

import matplotlib.pyplot as plt
ExperimentConfig = config.experiment_setup.ExperimentConfig
HyperparamOptManager = training.hyperparam_optimizer.HyperparamOptManager
ModelClass = core.omnitft_model.OmniTFT
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


def log_local_devices():
    from tensorflow.python.client import device_lib

    print("=== Local devices ===")
    for device in device_lib.list_local_devices():
        print(device)


def count_available_gpus():
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices and visible_devices != "-1":
        return len([device for device in visible_devices.split(",") if device.strip()])

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return max(1, len([line for line in result.stdout.splitlines() if line.strip()]))
    except Exception:
        return 1


def get_task_names(task_name=None):
    return [task_name] if task_name else list(FORMATTERS_TO_TRAIN)


def train_single_target(name, output_folder, use_gpu, gpu_id=0):
    print("\n" + "="*80)
    print(f"Starting training for: {name}")
    print("="*80 + "\n")

    config = ExperimentConfig(name, output_folder)
    formatter = config.make_data_formatter()

    result = train_task(
        expt_name=name,
        use_gpu=use_gpu,
        model_folder=os.path.join(config.model_folder, "fixed"),
        data_csv_path=config.data_csv_path,
        data_formatter=formatter,
        use_testing_mode=False,
        gpu_id=gpu_id)

    print(f"\n✓ Completed training for: {name}\n")
    print("Cleaning up resources...")
    tf.keras.backend.clear_session()
    tf.reset_default_graph()
    from components.embedding_layers import TFTDataCache
    TFTDataCache._data_cache.clear()
    import gc
    gc.collect()
    print("Resources cleaned.\n")

    return result


def launch_parallel_training(task_names, output_folder, use_gpu):
    script_path = os.path.realpath(__file__)
    script_dir = os.path.dirname(script_path)
    logs_root = os.path.join(
        script_dir,
        "logs",
        "parallel_{}".format(dte.datetime.now().strftime("%Y%m%d_%H%M%S")))
    os.makedirs(logs_root, exist_ok=True)

    available_gpus = count_available_gpus() if use_gpu else 0
    print("Available GPUs for parallel training: {}".format(available_gpus))

    processes = []
    try:
        for index, name in enumerate(task_names):
            gpu_id = index % max(1, available_gpus) if use_gpu else 0
            log_path = os.path.join(logs_root, "{}.log".format(name))
            log_handle = open(log_path, "w", buffering=1)

            cmd = [
                sys.executable,
                script_path,
                "." if output_folder is None else output_folder,
                "yes" if use_gpu else "no",
                "--task",
                name,
                "--gpu-id",
                str(gpu_id),
                "--skip-inference",
            ]

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            process = subprocess.Popen(
                cmd,
                cwd=script_dir,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT)

            processes.append((name, process, log_path, log_handle))
            print(
                f"Launched parallel worker for: {name} "
                f"(pid={process.pid}, gpu_id={gpu_id}, log={log_path})")

        for name, process, log_path, _ in processes:
            process.wait()
            if process.returncode != 0:
                raise RuntimeError(
                    f"Training failed for {name} (exit code {process.returncode}). "
                    f"See log: {log_path}")
            print(f"Worker {name} finished successfully. Log: {log_path}")
    finally:
        for _, _, _, log_handle in processes:
            log_handle.close()

    print("Per-task logs saved under {}".format(logs_root))
    return logs_root

def train_task(expt_name,
               use_gpu,
               model_folder,
               data_csv_path,
               data_formatter,
               use_testing_mode=False,
               gpu_id=0):

    num_repeats = 1

    if not isinstance(data_formatter, base_fmt.GenericDataFormatter):
        raise ValueError("Data formatters should inherit from GenericDataFormatter!")

    tf_config = utils.get_default_tensorflow_config(
        tf_device="gpu" if use_gpu else "cpu", gpu_id=gpu_id)

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

            if val_loss < best_loss:
                opt_manager.update_score(params, val_loss, model)
                best_loss = val_loss

    print("Training completed @ {}".format(dte.datetime.now()))
    print("Best validation loss = {}".format(best_loss))

    return {
        'task_name': expt_name,
        'best_params': opt_manager.get_best_params(),
        'model_folder': opt_manager.hyperparam_folder,
        'test_data': test,
        'valid_data': valid,
        'formatter': data_formatter,
    }


def _train_worker(name, output_folder, use_gpu, gpu_id=0):
    """Top-level worker function for multiprocessing parallel training.

    Each worker process trains one task independently, saves model to disk,
    then exits. Must be defined at module level for Windows multiprocessing.
    """
    import gc
    from components.embedding_layers import TFTDataCache

    print(f"\n[Worker {name}] Starting training on GPU {gpu_id}...")
    config = ExperimentConfig(name, output_folder)
    formatter = config.make_data_formatter()

    train_task(
        expt_name=name,
        use_gpu=use_gpu,
        model_folder=os.path.join(config.model_folder, "fixed"),
        data_csv_path=config.data_csv_path,
        data_formatter=formatter,
        use_testing_mode=False,
        gpu_id=gpu_id)

    print(f"[Worker {name}] Training completed. Cleaning up...")
    tf.keras.backend.clear_session()
    tf.reset_default_graph()
    TFTDataCache._data_cache.clear()
    gc.collect()


def _rebuild_trained_tasks(task_names, output_folder):
    """Rebuild trained_tasks dict from saved models after parallel training.

    Since split_data() is deterministic (fixed seed), re-splitting gives
    the exact same train/valid/test as during training.
    """
    import json

    trained_tasks = {}
    for name in task_names:
        config = ExperimentConfig(name, output_folder)
        formatter = config.make_data_formatter()
        model_folder = os.path.join(config.model_folder, "fixed")

        # Re-split data (deterministic - same seed gives same split)
        raw_data = pd.read_csv(config.data_csv_path, index_col=0)
        _, valid, test = formatter.split_data(raw_data)

        # Load best params from saved training result
        result_path = os.path.join(model_folder, 'best_result.json')
        with open(result_path, 'r') as f:
            best_params = json.load(f)['params']

        trained_tasks[name] = {
            'task_name': name,
            'best_params': best_params,
            'model_folder': model_folder,
            'test_data': test,
            'valid_data': valid,
            'formatter': formatter,
        }
    return trained_tasks


if __name__ == "__main__":
    def get_args():
        parser = argparse.ArgumentParser(description="Train TFT models for all targets")
        parser.add_argument("output_folder", metavar="output_folder", type=str, nargs="?", default=".",
                            help="Output folder path (default: current directory)")
        parser.add_argument("use_gpu", metavar="use_gpu", type=str, nargs="?", choices=["yes", "no"], default="yes",
                            help="Use GPU for training (yes/no, default: yes)")
        parser.add_argument("--parallel", action="store_true",
                            help="Train all tasks in parallel")
        parser.add_argument("--task", type=str, choices=FORMATTERS_TO_TRAIN,
                            help="Train only a single task")
        parser.add_argument("--gpu-id", type=int, default=0,
                            help="GPU id to use for single-task training")
        parser.add_argument("--skip-inference", action="store_true",
                            help="Skip multi-task inference after training")
        args = parser.parse_known_args()[0]
        root_folder = None if args.output_folder == "." else args.output_folder
        return (root_folder, args.use_gpu == "yes", args.parallel,
                args.task, args.gpu_id, args.skip_inference)

    (output_folder, use_tensorflow_with_gpu, use_parallel,
     task_name, gpu_id, skip_inference) = get_args()
    tasks_to_train = get_task_names(task_name)

    if use_parallel and task_name is not None:
        raise ValueError("--parallel cannot be combined with --task")

    if not use_parallel:
        log_local_devices()
    print("Using output folder {}".format(output_folder))
    print("Training targets: {}".format(", ".join(tasks_to_train)))
    print("Training mode: {}".format("PARALLEL" if use_parallel else "SEQUENTIAL"))

    if use_parallel:
        launch_parallel_training(tasks_to_train, output_folder, use_tensorflow_with_gpu)

        # Rebuild trained_tasks from saved models (data split is deterministic)
        trained_tasks = _rebuild_trained_tasks(tasks_to_train, output_folder)

    else:
        # Sequential: train one at a time (original behavior)
        trained_tasks = {}
        for name in tasks_to_train:
            result = train_single_target(
                name=name,
                output_folder=output_folder,
                use_gpu=use_tensorflow_with_gpu,
                gpu_id=gpu_id if (use_tensorflow_with_gpu and len(tasks_to_train) == 1) else 0)
            trained_tasks[name] = result

    if skip_inference:
        print("Skipping phase 2 inference as requested.")
        sys.exit(0)

    print("\n" + "="*80)
    print("Phase 2: Multi-task simultaneous inference")
    print("="*80 + "\n")

    from multi_task_predictor import MultiTaskPredictor
    predictor = MultiTaskPredictor(trained_tasks, use_tensorflow_with_gpu)
    combined_results = predictor.predict_all()
    predictor.print_metrics()

    if output_folder is None:
        results_root = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 'outputs', 'results')
    else:
        results_root = os.path.join(output_folder, 'results')
    predictor.save_results(results_root)
    predictor.close()

    print("\n" + "="*80)
    print("All tasks completed successfully!")
    print("="*80)
