# Lint as: python3
"""Classes used for hyperparameter optimisation.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import shutil
import libs.utils as utils
import numpy as np
import pandas as pd
import json

Deque = collections.deque


class HyperparamOptManager:

    def __init__(
        self, param_ranges, fixed_params, model_folder
    ):

        self.param_ranges = param_ranges
        self.fixed_params = fixed_params
        self.hyperparam_folder = model_folder
        utils.create_folder_if_not_exist(self.hyperparam_folder)
        self.results_filepath = os.path.join(self.hyperparam_folder, "best_result.json")
        
        # Generate all hyperparameter combinations
        self.all_combinations = self._generate_grid_search_combinations()
        self.current_index = 0
        print(f"Total hyperparameter combinations for Grid Search: {len(self.all_combinations)}")
        
        # Initialize best score and parameters
        self.best_score = np.Inf
        self.best_params = None
        
        # Initialize results dictionary to track all evaluated combinations
        self.results = {}
        
        # Load existing best result if exists
        if os.path.exists(self.results_filepath):
            self.load_best_result()

    def _generate_grid_search_combinations(self):
        """Generates all hyperparameter combinations for grid search without using itertools."""
        combinations = []
        keys = list(self.param_ranges.keys())
        values = list(self.param_ranges.values())
        total = len(values)
        
        def backtrack(index, current):
            if index == total:
                combinations.append(current.copy())
                return
            for value in values[index]:
                current[keys[index]] = value
                backtrack(index + 1, current)
        
        backtrack(0, {})
        return combinations

    def load_best_result(self):
        """Loads the best result from previous optimization."""
        with open(self.results_filepath, 'r') as f:
            try:
                best_result = json.load(f)
                self.best_score = best_result.get('loss', np.Inf)
                self.best_params = best_result.get('params', None)
                self.results = best_result.get('all_results', {})
                print(f"Loaded previous best score: {self.best_score} with params: {self.best_params}")
            except json.JSONDecodeError:
                print(f"Warning: {self.results_filepath} is empty or corrupted. Starting fresh.")
                self.best_score = np.Inf
                self.best_params = None
                self.results = {}

    def get_next_parameters(self):
        """Returns the next set of hyperparameters to optimize."""
        if self.current_index >= len(self.all_combinations):
            raise StopIteration("All hyperparameter combinations have been evaluated.")
        params = self.all_combinations[self.current_index]
        self.current_index += 1
        # Add fixed params
        for k in self.fixed_params:
            params[k] = self.fixed_params[k]
        return params

    def update_score(self, parameters, loss, model=None, info=""):
        """Updates the best score and parameters if current loss is better.
        """
        if np.isnan(loss):
            loss = np.Inf

        name = self._get_name(parameters)

        # Update results dictionary
        self.results[name] = {
            "loss": loss,
            "info": info,
            "params": parameters
        }

        # Check if current loss is better
        if loss < self.best_score:
            # Update best score and params
            self.best_score = loss
            # Extract only the hyperparameters (exclude fixed params)
            hyperparams = {k: v for k, v in parameters.items() if k in self.param_ranges}
            self.best_params = hyperparams
            print(f"New best score: {self.best_score} with params: {self.best_params}")
            
            # Save the model if required
            if model is not None:
                print("Saving the new best model...")
                model.save(self.hyperparam_folder)
            
        else:
            print(f"Current loss {loss} is not better than best loss {self.best_score}.")

        # Save all results to JSON
        best_result = {
            "loss": self.best_score,
            "params": self.best_params,
            "all_results": self.results
        }
        with open(self.results_filepath, 'w') as f:
            json.dump(best_result, f, indent=4)
        print(f"Results updated and saved to {self.results_filepath}")

        return loss < self.best_score

    def _check_params(self, params):
        """Checks that parameter map is properly defined."""

        valid_fields = list(self.param_ranges.keys()) + list(self.fixed_params.keys())
        invalid_fields = [k for k in params if k not in valid_fields]
        missing_fields = [k for k in valid_fields if k not in params]

        if invalid_fields:
            raise ValueError(
                "Invalid Fields Found {} - Valid ones are {}".format(
                    invalid_fields, valid_fields
                )
            )
        if missing_fields:
            raise ValueError(
                "Missing Fields Found {} - Valid ones are {}".format(
                    missing_fields, valid_fields
                )
            )

    def _get_name(self, params):
        """Returns a unique key for the supplied set of params."""

        self._check_params(params)

        fields = list(params.keys())
        fields.sort()

        return "_".join([str(params[k]) for k in fields])

    def get_best_params(self):
        """Returns the best hyperparameters found."""
        if self.best_params is None:
            raise ValueError("No hyperparameter combinations have been evaluated yet.")
        return self.best_params

    def clear(self):
        """Clears all previous best results."""
        if os.path.exists(self.results_filepath):
            os.remove(self.results_filepath)
            print(f"Removed existing best result at {self.results_filepath}.")
        self.best_score = np.Inf
        self.best_params = None
        self.current_index = 0
        self.results = {}
        print("Hyperparameter optimization manager has been reset.")


class DistributedHyperparamOptManager(HyperparamOptManager):
    """Manages distributed hyperparameter optimisation across many GPUs."""

    def __init__(
        self,
        param_ranges,
        fixed_params,
        root_model_folder,
        worker_number,
        search_iterations=1000,
        num_iterations_per_worker=5,
        clear_serialised_params=False,
    ):
        """Instantiates optimisation manager.

        This hyperparameter optimisation pre-generates #search_iterations
        hyperparameter combinations and serialises them
        at the start. At runtime, each worker goes through their own set of
        parameter ranges. The pregeneration
        allows for multiple workers to run in parallel on different machines without
        resulting in parameter overlaps.

        Args:
          param_ranges: Discrete hyperparameter range for grid search.
          fixed_params: Fixed model parameters per experiment.
          root_model_folder: Folder to store optimisation artifacts.
          worker_number: Worker index defining which set of hyperparameters to
            test.
          search_iterations: Total number of hyperparameter combinations to generate.
          num_iterations_per_worker: How many iterations are handled per worker.
          clear_serialised_params: Whether to regenerate hyperparameter
            combinations.
        """

        max_workers = int(np.ceil(search_iterations / num_iterations_per_worker))

        # Sanity checks
        if worker_number > max_workers:
            raise ValueError(
                "Worker number ({}) cannot be larger than the total number of workers!".format(
                    max_workers
                )
            )
        if worker_number > search_iterations:
            raise ValueError(
                "Worker number ({}) cannot be larger than the max search iterations ({})!".format(
                    worker_number, search_iterations
                )
            )

        print(
            "*** Creating hyperparameter manager for worker {} ***".format(
                worker_number
            )
        )

        hyperparam_folder = os.path.join(root_model_folder, str(worker_number))
        super().__init__(
            param_ranges, fixed_params, hyperparam_folder
        )

        serialised_ranges_folder = os.path.join(root_model_folder, "hyperparams")
        if clear_serialised_params:
            print("Regenerating hyperparameter list")
            if os.path.exists(serialised_ranges_folder):
                shutil.rmtree(serialised_ranges_folder)

        utils.create_folder_if_not_exist(serialised_ranges_folder)

        self.serialised_ranges_path = os.path.join(
            serialised_ranges_folder, "ranges_{}.json".format(search_iterations)
        )
        self.hyperparam_folder = hyperparam_folder  # override
        self.worker_num = worker_number
        self.total_search_iterations = search_iterations
        self.num_iterations_per_worker = num_iterations_per_worker
        self.global_hyperparam_df = self.load_serialised_hyperparam_df()
        self.worker_search_queue = self._get_worker_search_queue()

    @property
    def optimisation_completed(self):
        return False if self.worker_search_queue else True

    def get_next_parameters(self):
        """Returns next dictionary of hyperparameters to optimise."""
        if not self.worker_search_queue:
            raise StopIteration("No more hyperparameter combinations for this worker.")
        param_name = self.worker_search_queue.pop()

        params = self.global_hyperparam_df.loc[param_name, :].to_dict()

        # Always override!
        for k in self.fixed_params:
            print("Overriding saved {}: {}".format(k, self.fixed_params[k]))
            params[k] = self.fixed_params[k]

        return params

    def load_serialised_hyperparam_df(self):
        """Loads serialised hyperparameter ranges from file.

        Returns:
          DataFrame containing hyperparameter combinations.
        """
        print(
            "Loading params for {} search iterations from {}".format(
                self.total_search_iterations, self.serialised_ranges_path
            )
        )

        if os.path.exists(self.serialised_ranges_path):
            with open(self.serialised_ranges_path, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data).T
        else:
            print("Unable to load - regenerating search ranges instead")
            df = self.update_serialised_hyperparam_df()

        return df

    def update_serialised_hyperparam_df(self):

        search_df = self._generate_full_hyperparam_df()

        print(
            "Serialising params for {} search iterations to {}".format(
                self.total_search_iterations, self.serialised_ranges_path
            )
        )

        search_df.to_json(self.serialised_ranges_path, orient='index')

        return search_df

    def _generate_full_hyperparam_df(self):
        """Generates actual hyperparameter combinations.

        Returns:
          DataFrame containing hyperparameter combinations.
        """

        # For grid search, use the pre-generated all_combinations
        name_list = []
        param_list = []
        for i in range(min(self.total_search_iterations, len(self.all_combinations))):
            params = self.all_combinations[i]
            name = self._get_name(params)

            name_list.append(name)
            param_list.append(params)

        full_search_df = pd.DataFrame(param_list, index=name_list)

        return full_search_df

    def clear(self):  # reset when cleared
        """Clears results for hyperparameter manager and resets."""
        super().clear()
        self.worker_search_queue = self._get_worker_search_queue()

    def load_results(self):
        """Load results from file and queue parameter combinations to try.

        Returns:
          Boolean indicating if results were successfully loaded.
        """
        if os.path.exists(self.results_filepath):
            self.load_best_result()
            self.worker_search_queue = self._get_worker_search_queue()
            return True
        return False

    def _get_worker_search_queue(self):
        """Generates the queue of param combinations for current worker.

        Returns:
          Queue of hyperparameter combinations outstanding.
        """
        global_df = self.assign_worker_numbers(self.global_hyperparam_df)
        worker_df = global_df[global_df["worker"] == self.worker_num]

        # Exclude already evaluated parameter combinations
        left_overs = [s for s in worker_df.index if s not in self.results]
        return Deque(left_overs)

    def assign_worker_numbers(self, df):

        output = df.copy()

        n = self.total_search_iterations
        batch_size = self.num_iterations_per_worker

        max_worker_num = int(np.ceil(n / batch_size))

        worker_idx = np.concatenate(
            [
                np.tile(i + 1, self.num_iterations_per_worker)
                for i in range(max_worker_num)
            ]
        )

        output["worker"] = worker_idx[: len(output)]

        return output
