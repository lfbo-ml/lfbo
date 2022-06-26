import ConfigSpace as CS

from hyperopt import hp
from hyperopt.pyll.base import scope

def config_space_to_search_space(config_space, q=1):
    """
    Converts HpBandSter's ConfigurationSpace to HyperOpt's search space
    dictionary format.
    """
    search_space = {}
    for h in config_space.get_hyperparameters():
        if isinstance(h, CS.OrdinalHyperparameter):
            search_space[h.name] = hp.quniform(h.name, 0, len(h.sequence)-1, q)
        elif isinstance(h, CS.CategoricalHyperparameter):
            search_space[h.name] = hp.choice(h.name, h.choices)
        elif isinstance(h, CS.UniformIntegerHyperparameter):
            search_space[h.name] = scope.int(hp.quniform(h.name, h.lower, h.upper, q))
        elif isinstance(h, CS.UniformFloatHyperparameter):
            search_space[h.name] = hp.uniform(h.name, h.lower, h.upper)

    return search_space


def config_space_to_domain(config_space):
    """
    Converts HpBandSter's ConfigurationSpace to GPyOpt's domain list of dicts
    format.
    """
    space = []
    for h in config_space.get_hyperparameters():

        if isinstance(h, CS.OrdinalHyperparameter):
            d = dict(name=h.name, type="discrete",
                     domain=(0, len(h.sequence) - 1))
        elif isinstance(h, CS.CategoricalHyperparameter):
            d = dict(name=h.name, type="categorical",
                     domain=[i for i, _ in enumerate(h.choices)])
        elif isinstance(h, CS.UniformIntegerHyperparameter):
            d = dict(name=h.name, type="discrete", domain=(h.lower, h.upper))
        elif isinstance(h, CS.UniformFloatHyperparameter):
            d = dict(name=h.name, type="continuous", domain=(h.lower, h.upper),
                     dimensionality=1)
        space.append(d)
    return space


def kwargs_to_config(kwargs, config_space):

    config = {}
    for h in config_space.get_hyperparameters():
        if isinstance(h, CS.OrdinalHyperparameter):
            value = h.sequence[int(kwargs[h.name])]
        elif isinstance(h, CS.UniformIntegerHyperparameter):
            value = int(kwargs[h.name])
        else:
            value = kwargs[h.name]
        config[h.name] = value

    return config
