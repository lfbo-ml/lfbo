import numpy as np
import os
from .base import Benchmark, Evaluation
import ConfigSpace as CS
from nas_201_api import NASBench201API as API


class NasBench201(Benchmark):
    def __init__(self, dataset_name, input_dir):
        self.INPUT = 'input'
        self.OUTPUT = 'output'
        self.OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
        self.NUM_OPS = len(self.OPS)
        self.OP_SPOTS = 6
        self.dataset = dataset_name
        self.nasbench = API(os.path.join(input_dir, 'NAS-Bench-201-v1_1-096897.pth'))

    def evaluate(self, config, budget=None):
        deterministic = True  # True if used for getting the minimum mean validation error
        ops = []
        for i in range(self.OP_SPOTS):
            ops.append(config[f"op_{i}"])
        string = self.get_string_from_ops(ops)
        index = self.nasbench.query_index_by_arch(string)
        results = self.nasbench.query_by_index(index, self.dataset)
        accs = []
        times = []
        for key in results.keys():
            accs.append(results[key].get_eval('x-valid')['accuracy'])
            times.append(results[key].get_eval('x-valid')['all_time'])
        if not deterministic:
            sample_id = np.random.choice(len(accs))
            loss = round(100 - accs[sample_id], 10) / 100.
            time = times[sample_id]
            return Evaluation(value=loss, duration=time)
        else:
            loss = round(100 - np.mean(accs), 10) / 100.
            time = np.mean(times)
            return Evaluation(value=loss, duration=time)

    def get_string_from_ops(self, ops):
        # given a list of operations, get the string
        strings = ['|']
        nodes = [0, 0, 1, 0, 1, 2]
        for i, op in enumerate(ops):
            strings.append(op + '~{}|'.format(nodes[i]))
            if i < len(nodes) - 1 and nodes[i + 1] == 0:
                strings.append('+|')
        return ''.join(strings)

    def get_config_space(self):
        cs = CS.ConfigurationSpace()
        for i in range(self.OP_SPOTS):
            cs.add_hyperparameter(CS.CategoricalHyperparameter(f"op_{i}", self.OPS))
        return cs

    def get_minimum(self):
        # minimum_loss = np.inf
        # cs = self.get_config_space()
        # x = cs.sample_configuration()
        # for op0 in self.OPS:
        #     x['op_0'] = op0
        #     for op1 in self.OPS:
        #         x['op_1'] = op1
        #         for op2 in self.OPS:
        #             x['op_2'] = op2
        #             for op3 in self.OPS:
        #                 x['op_3'] = op3
        #                 for op4 in self.OPS:
        #                     x['op_4'] = op4
        #                     for op5 in self.OPS:
        #                         x['op_5'] = op5
        #                         result = self.evaluate(x)
        #                         minimum_loss = min(minimum_loss, result.value)
        # print("Minimum mean validation error", minimum_loss)
        # return minimum_loss
        if self.dataset == 'cifar10-valid':
            return 0.151080000098
        elif self.dataset == 'cifar100':
            return 0.3868
        elif self.dataset == 'ImageNet16-120':
            return 0.61333333374
        else:
            raise NotImplementedError
