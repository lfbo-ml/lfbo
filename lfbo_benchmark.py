import os
from argparse import ArgumentParser
from benchmarks import make_benchmark
from model import LFBO
import pickle
import numpy as np

BENCHMARK_CHOICES = ['fcnet_alt', 'nasbench201']
DATASET_CHOICES = [
    'naval', 'parkinsons', 'protein', 'slice',  # for fcnet_alt
    'cifar10', 'cifar100', 'imagenet'  # for nasbench201
]
WEIGHT_TYPE_CHOICES = ['pi', 'ei']  # ei: Expected Improvement, pi: Probability of Improvement
MODEL_TYPE_CHOICES = ['mlp', 'rf', 'xgb']  # mlp: MLP, rf: Random Forest, xgb: XGBoost Classifier
DATASET_DIR = "datasets"  # Your dataset directory


def run(args):
    benchmark = make_benchmark(benchmark_name=args.benchmark, dataset_name=args.dataset, input_dir=DATASET_DIR)

    for seed in range(args.start_seed, args.end_seed + 1):
        lfbo = LFBO(config_space=benchmark.get_config_space(),
                    num_random_init=args.num_random_init,
                    gamma=args.gamma,
                    weight_type=args.weight_type,
                    model_type=args.model_type,
                    nasbench=args.nasbench,
                    seed=seed)

        losses = []
        for i in range(args.iterations):
            config_dict = lfbo.step()
            evaluation = benchmark.evaluate(config_dict)
            loss = evaluation.value
            lfbo.add_new_observation(config_dict, loss)
            losses.append(loss)
            print(f"Seed: {seed} \t Step {i} \t Func of the explored x: {loss} \t Best so far: {np.min(losses)}")

        save_path = f'results/{args.benchmark}-{args.dataset}/{args.model_type}-{args.weight_type}'
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f'{seed}.pkl'), 'wb') as fout:
            pickle.dump(losses, fout)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--benchmark", type=str, choices=BENCHMARK_CHOICES)
    parser.add_argument("--dataset", type=str, choices=DATASET_CHOICES)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--num_random_init", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.33)
    parser.add_argument("--weight_type", type=str, choices=WEIGHT_TYPE_CHOICES)
    parser.add_argument("--model_type", type=str, choices=MODEL_TYPE_CHOICES)
    parser.add_argument("--start_seed", type=int, default=0)
    parser.add_argument("--end_seed", type=int, default=100)
    args = parser.parse_args()
    if 'nasbench' in args.benchmark:
        args.nasbench = True
    else:
        args.nasbench = False

    run(args)
