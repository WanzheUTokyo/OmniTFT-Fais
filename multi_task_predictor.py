

import gc
import os

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import core.omnitft_model
import config.experiment_setup
import training.training_utils as utils

ModelClass = core.omnitft_model.OmniTFT
ExperimentConfig = config.experiment_setup.ExperimentConfig


class MultiTaskPredictor:
    """Holds all task models simultaneously and predicts all tasks at once."""

    def __init__(self, task_configs, use_gpu):
        """Load all trained models into separate TF Graphs.

        Args:
            task_configs: dict of {task_name: {
                'best_params': dict,
                'model_folder': str,
                'test_data': DataFrame,
                'valid_data': DataFrame,
                'formatter': GenericDataFormatter,
            }}
            use_gpu: bool
        """
        self._tasks = {}
        self._results = {}
        self._use_gpu = use_gpu
        self._tf_config = utils.get_default_tensorflow_config(
            tf_device="gpu" if use_gpu else "cpu", gpu_id=0)

        print("Loading all task models simultaneously...")
        for name, cfg in task_configs.items():
            print(f"  Loading model for: {name}")
            graph = tf.Graph()
            with graph.as_default():
                sess = tf.compat.v1.Session(config=self._tf_config)
                tf.keras.backend.set_session(sess)
                model = ModelClass(cfg['best_params'], use_cudnn=use_gpu)
                model.load(cfg['model_folder'])

            self._tasks[name] = {
                'model': model,
                'formatter': cfg['formatter'],
                'test_data': cfg['test_data'],
                'valid_data': cfg['valid_data'],
                'graph': graph,
                'session': sess,
            }
        print(f"All {len(self._tasks)} models loaded simultaneously.\n")

    @classmethod
    def load_from_saved(cls, model_root, tasks, use_gpu=True):
        """Load a MultiTaskPredictor from previously saved models.

        This is for standalone inference without re-training.

        Args:
            model_root: path to saved_models directory
                        (e.g., 'outputs/saved_models')
            tasks: list of task names (e.g., ['Creatinine', 'Lactate'])
            use_gpu: bool

        Returns:
            MultiTaskPredictor instance with all models loaded.
        """
        task_configs = {}
        for name in tasks:
            model_folder = os.path.join(model_root, name, 'fixed')
            scaler_folder = os.path.join(model_folder, 'scalers')

            exp_config = ExperimentConfig(name,
                                          os.path.dirname(
                                              os.path.dirname(model_root)))
            formatter = exp_config.make_data_formatter()
            formatter.load_and_set_scalers(model_folder)

            # Load best params from saved result
            import json
            result_path = os.path.join(model_folder, 'best_result.json')
            with open(result_path, 'r') as f:
                best_result = json.load(f)
            best_params = best_result['params']

            task_configs[name] = {
                'best_params': best_params,
                'model_folder': model_folder,
                'test_data': None,
                'valid_data': None,
                'formatter': formatter,
            }

        return cls(task_configs, use_gpu)

    def predict_all(self, test_data_override=None):
        """Run all models and return per-task predictions.

        Args:
            test_data_override: optional dict {task_name: DataFrame}.
                Use this to provide custom test data instead of the data
                from training. Required when using load_from_saved().

        Returns:
            dict: {task_name: {targets, p10, p50, p90, metrics}}
        """
        print("*** Running multi-task simultaneous inference ***")
        self._results = {}

        for name, task in self._tasks.items():
            test_data = (test_data_override or {}).get(name, task['test_data'])
            if test_data is None:
                print(f"  Skipping {name}: no test data provided.")
                continue

            print(f"  Predicting: {name}")
            with task['graph'].as_default():
                tf.keras.backend.set_session(task['session'])
                output_map = task['model'].predict(
                    test_data, return_targets=True)

                fmt = task['formatter']
                targets = fmt.format_predictions(output_map['targets'])
                p10 = fmt.format_predictions(output_map['p10'])
                p50 = fmt.format_predictions(output_map['p50'])
                p90 = fmt.format_predictions(output_map['p90'])

                def strip(df):
                    return df[[c for c in df.columns
                               if c not in {"forecast_time", "identifier"}]]

                p10_loss = utils.numpy_normalised_quantile_loss(
                    strip(targets), strip(p10), 0.1)
                p50_loss = utils.numpy_normalised_quantile_loss(
                    strip(targets), strip(p50), 0.5)
                p90_loss = utils.numpy_normalised_quantile_loss(
                    strip(targets), strip(p90), 0.9)

            self._results[name] = {
                'targets': targets,
                'p10': p10,
                'p50': p50,
                'p90': p90,
                'p10_loss': p10_loss.mean(),
                'p50_loss': p50_loss.mean(),
                'p90_loss': p90_loss.mean(),
            }

        print("Multi-task inference completed.\n")
        return self._results

    def get_combined_predictions(self, quantile='p50'):

        if not self._results:
            raise ValueError(
                "No predictions available. Call predict_all() first.")

        dfs = []
        for name, result in self._results.items():
            pred_df = result[quantile].copy()
            time_cols = [c for c in pred_df.columns
                         if c not in {"forecast_time", "identifier"}]
            rename_map = {c: f"{name}_{c}" for c in time_cols}
            pred_df = pred_df.rename(columns=rename_map)
            dfs.append(pred_df)

        if not dfs:
            return pd.DataFrame()

        combined = dfs[0]
        merge_cols = [c for c in ["identifier", "forecast_time"]
                      if c in combined.columns]
        for df in dfs[1:]:
            combined = pd.merge(combined, df, on=merge_cols, how="outer")

        return combined

    def print_metrics(self):

        if not self._results:
            print("No results available. Call predict_all() first.")
            return

        print("=" * 60)
        print("Multi-Task Prediction Metrics")
        print("=" * 60)
        for name, result in self._results.items():
            print(f"\n  [{name}]")
            print(f"    P10 Loss: {result['p10_loss']:.6f}")
            print(f"    P50 Loss: {result['p50_loss']:.6f}")
            print(f"    P90 Loss: {result['p90_loss']:.6f}")
        print("\n" + "=" * 60)

    def save_results(self, results_root):

        if not self._results:
            print("No results to save. Call predict_all() first.")
            return

        # Save per-task results
        for name, result in self._results.items():
            task_folder = os.path.join(results_root, name)
            os.makedirs(task_folder, exist_ok=True)
            for key in ['targets', 'p10', 'p50', 'p90']:
                filepath = os.path.join(task_folder, f"{key}.csv")
                result[key].to_csv(filepath, index=False)
            print(f"  Saved {name} predictions to {task_folder}")

        # Save combined predictions (all tasks merged by patient)
        combined_folder = os.path.join(results_root, "combined")
        os.makedirs(combined_folder, exist_ok=True)
        for q in ['p10', 'p50', 'p90']:
            combined = self.get_combined_predictions(quantile=q)
            filepath = os.path.join(combined_folder, f"combined_{q}.csv")
            combined.to_csv(filepath, index=False)
        print(f"  Saved combined predictions to {combined_folder}")

    def close(self):
        """Clean up all TF sessions and free memory."""
        for name, task in self._tasks.items():
            try:
                task['session'].close()
            except Exception:
                pass
        self._tasks.clear()
        self._results.clear()
        gc.collect()
        print("MultiTaskPredictor resources cleaned.")
