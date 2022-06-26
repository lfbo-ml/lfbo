import os.path
import pickle
import numpy as np
from benchmarks import make_benchmark
import matplotlib.pyplot as plt


# benchmark_name = 'fcnet_alt'
# dataset_name = 'slice'
# method_list = ['mlp-ei', 'mlp-pi', 'rf-ei', 'rf-pi', 'xgb-ei', 'xgb-pi']

benchmark_name = 'nasbench201'
dataset_name = 'imagenet'
method_list = ['mlp-ei', 'mlp-pi', 'rf-ei', 'rf-pi', 'xgb-ei', 'xgb-pi']

num_runs = 100

nasbench201_minimum = {'cifar10': 0.151080000098, 'cifar100': 0.3868, 'imagenet': 0.61333333374}
if benchmark_name == 'nasbench201':
    loss_min = nasbench201_minimum[dataset_name]
else:
    DATASET_DIR = "datasets"
    benchmark = make_benchmark(benchmark_name=benchmark_name, dataset_name=dataset_name, input_dir=DATASET_DIR)
    loss_min = benchmark.get_minimum()
print("Minimum loss", loss_min)

min_clip_value = {'naval': 1e-8, 'slice': 1e-8, 'parkinsons': 1e-7, 'protein': 1e-7,
                  'cifar10': -np.inf, 'cifar100': -np.inf, 'imagenet': -np.inf}

regret_list = []
for method in method_list:
    path = os.path.join("results", f"{benchmark_name}-{dataset_name}", method)
    losses = []
    for i in range(num_runs):
        data = pickle.load(open(os.path.join(path, f"{i}.pkl"), 'rb'))
        loss = [np.min(data[:j]) for j in range(1, 1 + len(data))]
        losses.append(loss)
    losses = np.array(losses)
    loss_mean = np.mean(losses, axis=0)
    loss_std = np.std(losses, axis=0)
    regret = loss_mean - loss_min
    regret = np.clip(regret, a_min=min_clip_value[dataset_name], a_max=np.inf)
    regret_list.append(regret)

method_to_label = {'mlp-ei': 'LFBO (MLP)', 'mlp-pi': 'BORE (MLP)',
                   'rf-ei': 'LFBO (RF)', 'rf-pi': 'BORE (RF)',
                   'xgb-ei': 'LFBO (XGB)', 'xgb-pi': 'BORE (XGB)'}

plt.figure()
for i, method in enumerate(method_list):
    x = list(range(len(regret_list[i])))
    plt.plot(x, regret_list[i], label=method_to_label[method])
plt.legend()
plt.title(f"{benchmark_name}-{dataset_name}")
plt.tight_layout()
plt.yscale('log')
if benchmark_name == 'fcnet_alt':
    plt.ylim(bottom=min_clip_value[dataset_name])
if dataset_name == 'slice':
    plt.xlim(right=150)
save_path = os.path.join("figures", f"{benchmark_name}-{dataset_name}.png")
os.makedirs("figures", exist_ok=True)
plt.savefig(save_path)
print(f"Save to {save_path}")
