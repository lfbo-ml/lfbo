from .synthetic import (Ackley, Branin, GoldsteinPrice, Rosenbrock,
                        SixHumpCamel, StyblinskiTang, Michalewicz, Hartmann3D,
                        Hartmann6D, Forrester, Bliznyuk)
from .tabular import FCNet, FCNetAlt
from .nas_benchmarks import NasBench201
from .surrogate import BOHBSurrogate

benchmarks = dict(
    forrester=Forrester,
    branin=Branin,
    goldstein_price=GoldsteinPrice,
    six_hump_camel=SixHumpCamel,
    ackley=Ackley,
    rosenbrock=Rosenbrock,
    styblinski_tang=StyblinskiTang,
    michalewicz=Michalewicz,
    hartmann3d=Hartmann3D,
    hartmann6d=Hartmann6D,
    fcnet=FCNet,
    fcnet_alt=FCNetAlt,
    bohb_surrogate=BOHBSurrogate,
    bliznyuk=Bliznyuk,
    nasbench201=NasBench201
)


def make_benchmark(benchmark_name, dimensions=None, dataset_name=None, input_dir="datasets/"):

    Benchmark = benchmarks[benchmark_name]

    kws = {}
    if any(map(benchmark_name.startswith, ("fcnet", "bohb_surrogate"))):
        assert dataset_name is not None, "must specify dataset name"
        assert input_dir is not None, "must specify data directory"
        kws["dataset_name"] = dataset_name
        kws["input_dir"] = input_dir

    if benchmark_name in ("michalewicz", "styblinski_tang", "rosenbrock", "ackley"):
        assert dimensions is not None, "must specify dimensions"
        kws["dimensions"] = dimensions

    if benchmark_name == 'nasbench201':
        kws["input_dir"] = input_dir
        assert dataset_name is not None, "must specify dataset name"
        if dataset_name == 'cifar10':
            kws["dataset_name"] = 'cifar10-valid'
        elif dataset_name == 'imagenet':
            kws["dataset_name"] = 'ImageNet16-120'
        else:
            kws["dataset_name"] = dataset_name
    return Benchmark(**kws)
