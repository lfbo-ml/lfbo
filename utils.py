import tensorflow as tf
from functools import wraps
from scipy.optimize import Bounds
import numpy as np
from scipy.stats import truncnorm
import ConfigSpace as CS


def stack(fn):
    @wraps(fn)
    def new_fn(*args):
        return fn(tf.stack(args))

    return new_fn


def unstack(fn):
    @wraps(fn)
    def new_fn(args):
        return fn(*tf.unstack(args, axis=-1))

    return new_fn


def squeeze(axis):
    def squeeze_dec(fn):
        @wraps(fn)
        def new_fn(*args, **kwargs):
            return tf.squeeze(fn(*args, **kwargs), axis=axis)

        return new_fn

    return squeeze_dec


def unbatch(fn):
    @wraps(fn)
    def new_fn(input):
        batch_input = tf.expand_dims(input, axis=0)
        batch_output = fn(batch_input)
        return tf.squeeze(batch_output, axis=0)

    return new_fn


def value_and_gradient(value_fn):
    @wraps(value_fn)
    @tf.function
    def value_and_gradient_fn(x):
        # Equivalent to `tfp.math.value_and_gradient(value_fn, x)`, with the
        # only difference that the gradients preserve their `dtype` rather than
        # casting to `tf.float32`, which is problematic for scipy.optimize
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(x)
            val = value_fn(x)

        grad = tape.gradient(val, x)

        return val, grad

    return value_and_gradient_fn


def numpy_io(fn):
    @wraps(fn)
    def new_fn(*args):
        new_args = map(tf.convert_to_tensor, args)
        outputs = fn(*new_args)
        new_outputs = [output.numpy() for output in outputs]

        return new_outputs

    return new_fn


def convert(model, transform=tf.identity):
    """
    Given a Keras model, builds a callable that takes a single array as input
    (rather than a batch of Tensors) and returns a pair containing the output
    value (a scalar) and the gradient vector (an array).

    This function makes it easy to use optimization methods
    from ``scipy.optimize`` to minimize inputs to a model wrt to its output
    using the option ``jac=True``.

    Parameters
    ----------
    model : a Keras model
        A Keras model, or any batched TensorFlow operation, with output
        dimension 1. More precisely, any operation that takes a Tensor of
        shape ``(None, D)`` as input and returns as output a Tensor of
        shape ``(None, 1)``.
    transform : callable, optional
        A function that transforms the output of the model, e.g. negating the
        output effectively maximizes instead of minimizes it.

    Returns
    -------
    fn : callable
        A function that takes an array of shape ``(D,)`` as input, and returns
        a pair with shape ``(), (D,)``, consisting of the output scalar and the
        gradient vector.
    """

    @numpy_io  # array input to Tensor and Tensor outputs back to array
    @value_and_gradient  # `(D,) -> ()` to `(D,) -> (), (D,)`
    @squeeze(axis=-1)  # `(D,) -> (1,)` to `(D,) -> ()`
    @unbatch  # `(None, D) -> (None, 1)` to `(D,) -> (1,)`
    def fn(x):
        return transform(model(x))

    return fn


def from_bounds(bounds):
    if isinstance(bounds, Bounds):
        low = bounds.lb
        high = bounds.ub
        dim = len(low)
        assert dim == len(high), "lower and upper bounds sizes do not match!"
    else:
        # assumes `bounds` is a list of tuples
        low, high = zip(*bounds)
        dim = len(bounds)

    return (low, high), dim


def ceil_divide(a, b, *args, **kwargs):
    return - np.floor_divide(-a, b, *args, **kwargs)


def steps_per_epoch(dataset_size, batch_size):
    return int(ceil_divide(dataset_size, batch_size))


def truncated_normal(loc, scale, lower, upper):
    a = (lower - loc) / scale
    b = (upper - loc) / scale
    return truncnorm(a=a, b=b, loc=loc, scale=scale)


def maybe_distort(loc, distortion=None, bounds=None, random_state=None,
                  print_fn=print):
    if distortion is None:
        return loc

    assert bounds is not None, "must specify bounds!"
    ret = truncated_normal(loc=loc,
                           scale=distortion,
                           lower=bounds.lb,
                           upper=bounds.ub).rvs(random_state=random_state)
    print_fn(f"Suggesting x={ret} (after applying distortion={distortion:.3E})")

    return ret


class DenseConfigurationSpace(CS.ConfigurationSpace):

    def __init__(self, other, *args, **kwargs):

        super(DenseConfigurationSpace, self).__init__(*args, **kwargs)
        # deep-copy only the hyperparameters. conditions, clauses, seed,
        # and other metadata ignored
        self.add_hyperparameters(other.get_hyperparameters())

        nums, cats, size_sparse, size_dense = self._get_mappings()

        if nums:
            self.num_src, self.num_trg = map(np.uintp, zip(*nums))

        if cats:
            self.cat_src, self.cat_trg, self.cat_sizes = \
                map(np.uintp, zip(*cats))

        self.nums = nums
        self.cats = cats
        self.size_sparse = size_sparse
        self.size_dense = size_dense

    def get_dimensions(self, sparse=False):
        return self.size_sparse if sparse else self.size_dense

    def sample_configuration(self, size=1):

        config_sparse = super(DenseConfigurationSpace, self) \
            .sample_configuration(size=size)

        configs_sparse_list = config_sparse if size > 1 else [config_sparse]

        configs = []
        for config in configs_sparse_list:
            configs.append(DenseConfiguration(self, values=config.get_dictionary()))

        return configs if size > 1 else configs.pop()

    def get_bounds(self):
        lowers = np.zeros(self.size_dense)
        uppers = np.ones(self.size_dense)

        # return list(zip(lowers, uppers))
        return Bounds(lowers, uppers)

    def _get_mappings(self):

        nums = []
        cats = []

        src_ind = trg_ind = 0
        for src_ind, hp in enumerate(self.get_hyperparameters()):
            if isinstance(hp, CS.CategoricalHyperparameter):
                cat_size = hp.num_choices
                cats.append((src_ind, trg_ind, cat_size))
                trg_ind += cat_size
            elif isinstance(hp, (CS.UniformIntegerHyperparameter,
                                 CS.UniformFloatHyperparameter)):
                nums.append((src_ind, trg_ind))
                trg_ind += 1
            else:
                raise NotImplementedError(
                    "Only hyperparameters of types "
                    "`CategoricalHyperparameter`, "
                    "`UniformIntegerHyperparameter`, "
                    "`UniformFloatHyperparameter` are supported!")

        size_sparse = src_ind + 1
        size_dense = trg_ind

        return nums, cats, size_sparse, size_dense


class DenseConfiguration(CS.Configuration):

    def __init__(self, configuration_space, *args, **kwargs):

        assert isinstance(configuration_space, DenseConfigurationSpace)
        super(DenseConfiguration, self).__init__(configuration_space,
                                                 *args, **kwargs)

    @classmethod
    def from_array(cls, configuration_space, array_dense, dtype="float64"):

        assert isinstance(configuration_space, DenseConfigurationSpace)
        cs = configuration_space
        # initialize output array
        array_sparse = np.empty(cs.size_sparse, dtype=dtype)

        # process numerical hyperparameters
        if cs.nums:
            array_sparse[cs.num_src] = array_dense[cs.num_trg]

        # process categorical hyperparameters
        for src_ind, trg_ind, size in cs.cats:
            ind_max = np.argmax(array_dense[trg_ind:trg_ind + size])
            array_sparse[src_ind] = ind_max

        return cls(configuration_space=configuration_space, vector=array_sparse)

    def to_array(self, dtype="float64"):

        cs = self.configuration_space
        array_sparse = super(DenseConfiguration, self).get_array()

        # initialize output array
        # TODO(LT): specify `dtype` flexibly
        array_dense = np.zeros(cs.size_dense, dtype=dtype)

        # process numerical hyperparameters
        if cs.nums:
            array_dense[cs.num_trg] = array_sparse[cs.num_src]

        # process categorical hyperparameters
        if cs.cats:
            cat_trg_offset = np.uintp(array_sparse[cs.cat_src])
            array_dense[cs.cat_trg + cat_trg_offset] = 1

        return array_dense


def dict_from_array(config_space, array):
    config = DenseConfiguration.from_array(config_space, array_dense=array)
    return config.get_dictionary()


def array_from_dict(config_space, dct):
    config = DenseConfiguration(config_space, values=dct)
    return config.to_array()
