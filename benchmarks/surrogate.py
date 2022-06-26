import numpy as np
import pickle
import ConfigSpace as CS

from .base import Benchmark, Evaluation

from pathlib import Path


DATASET_NAMES = ["adult", "higgs", "letter", "mnist", "optdigits", "poker",
                 "vehicle"]


class BOHBSurrogate(Benchmark):

    def __init__(self, dataset_name, input_dir):
        assert dataset_name in DATASET_NAMES, \
            f"`dataset_name` must be one of {DATASET_NAMES}!"
        base_path = Path(input_dir).joinpath("surrogates")
        obj_path = base_path.joinpath(f"rf_surrogate_paramnet_{dataset_name}.pkl")
        cost_path = base_path.joinpath(f"rf_cost_surrogate_paramnet_{dataset_name}.pkl")
        with open(obj_path, 'rb') as fh:
            self.surrogate_objective = pickle.load(fh)
        with open(cost_path, 'rb') as fh:
            self.surrogate_costs = pickle.load(fh)

    def evaluate(self, kwargs, budget=50):

        epoch = int(budget)

        x = np.array([
            10 ** kwargs['initial_lr_log10'],
            round(2 ** kwargs['batch_size_log2']),
            round(2 ** kwargs['average_units_per_layer_log2']),
            10 ** kwargs['final_lr_fraction_log2'],
            kwargs['shape_parameter_1'],
            kwargs['num_layers'],
            kwargs['dropout_0'],
            kwargs['dropout_1']
        ])

        X = np.atleast_2d(x)

        lc = self.surrogate_objective.predict(X).squeeze(axis=0)
        total_epochs = len(lc)

        c = self.surrogate_costs.predict(X).squeeze(axis=0)
        time_per_epoch = c / total_epochs

        value = lc[epoch-1]
        duration = time_per_epoch * epoch

        return Evaluation(value=value, duration=duration)

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        cs.add_hyperparameters([
            CS.UniformFloatHyperparameter('initial_lr_log10', lower=-6, upper=-2),
            CS.UniformFloatHyperparameter('batch_size_log2', lower=3, upper=8),
            CS.UniformFloatHyperparameter('average_units_per_layer_log2', lower=4, upper=8),
            CS.UniformFloatHyperparameter('final_lr_fraction_log2', lower=-4, upper=0),
            CS.UniformFloatHyperparameter('shape_parameter_1', lower=0., upper=1.),
            CS.UniformIntegerHyperparameter('num_layers', lower=1, upper=5),
            CS.UniformFloatHyperparameter('dropout_0', lower=0., upper=0.5),
            CS.UniformFloatHyperparameter('dropout_1', lower=0., upper=0.5),
        ])
        return cs

    def get_minimum(self):
        return 0.
        # raise NotImplementedError
