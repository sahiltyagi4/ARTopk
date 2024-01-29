import torch
import numpy as np
import random
import math
from _random import Random
from prettytable import PrettyTable


def set_seed(seed, determinism=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    rng = Random()
    rng.seed(seed)
    torch.use_deterministic_algorithms(determinism)


class AverageMeter(object):
    """Computes and stores the average and current value for model loss, accuracy etc."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EWMAMeter(object):
    def __init__(self, windowsize=5, alpha=0.01):
        self.windowsize = windowsize
        self.alpha = alpha
        self.window = []
        self.ewma_val = 0.

    def smoothendata(self, val):
        sum_val = 0.
        count = 0.
        self.window.append(val)
        if len(self.window) == self.windowsize:
            for index, datum in reversed(list(enumerate(self.window))):
                pow_ix = len(self.window) - index
                sum_val += math.pow((1 - self.alpha), pow_ix) * datum
                count += math.pow((1 - self.alpha), pow_ix)

            self.ewma_val = sum_val / count
            self.window.pop(0)

    def smooth_val(self):
        return self.ewma_val


class ModelStatistics(object):
    def __init__(self, param_names, param_shapes, param_lengths, windowsize, smoothing):
        self.param_names = param_names
        self.param_shapes = param_shapes
        self.param_lengths = param_lengths
        self.smooth_meter = EWMAMeter(windowsize=windowsize, alpha=smoothing)

        table = PrettyTable(["Module", "Parameters"])
        total_params = 0
        ctr = 0
        for name, p in param_shapes.items():
            param = param_lengths[name]
            table.add_row([name, param])
            total_params += param
            ctr += 1

        print(table)
        print(f"counter is {ctr}")
        print(f"Total Trainable Params: {total_params}")
        total_size = (total_params * 4) / (1024 * 1024)
        print(f"Model memory footprint using single precision: {total_size} MB")

    def compute_gradnorm(self, gradients):
        total_gnorm = 0.
        for g in gradients:
            total_gnorm += torch.norm(g.flatten())

        return total_gnorm

    def compute_imgaccuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def initialize_latency_bandwidth(model_name):
    latency, bw = [], []
    if model_name == 'resnet50':
        totalepochs = 101
        for i in range(0, totalepochs):
            if 0 <= i < 8:
                latency.append(1)
                bw.append(25)
            elif 8 <= i < 16:
                latency.append(2)
                bw.append(20)
            elif 16 <= i < 24:
                latency.append(4)
                bw.append(15)
            elif 24 <= i < 32:
                latency.append(8)
                bw.append(10)
            elif 32 <= i < 40:
                latency.append(16)
                bw.append(5)
            elif 40 <= i < 48:
                latency.append(32)
                bw.append(1)
            elif 48 <= i < 56:
                latency.append(32)
                bw.append(1)
            elif 56 <= i < 64:
                latency.append(16)
                bw.append(5)
            elif 64 <= i < 72:
                latency.append(8)
                bw.append(10)
            elif 72 <= i < 80:
                latency.append(4)
                bw.append(15)
            elif 80 <= i < 88:
                latency.append(2)
                bw.append(20)
            elif 88 <= i < 96:
                latency.append(1)
                bw.append(25)
            else:
                latency.append(1)
                bw.append(25)
    else:
        # for resnet18, alexnet and vit
        totalepochs = 51
        for i in range(0, totalepochs):
            if 0 <= i < 4:
                latency.append(1)
                bw.append(25)
            elif 4 <= i < 8:
                latency.append(2)
                bw.append(20)
            elif 8 <= i < 12:
                latency.append(4)
                bw.append(15)
            elif 12 <= i < 16:
                latency.append(8)
                bw.append(10)
            elif 16 <= i < 20:
                latency.append(16)
                bw.append(5)
            elif 20 <= i < 24:
                latency.append(32)
                bw.append(1)
            elif 24 <= i < 28:
                latency.append(32)
                bw.append(1)
            elif 28 <= i < 32:
                latency.append(16)
                bw.append(5)
            elif 32 <= i < 36:
                latency.append(8)
                bw.append(10)
            elif 36 <= i < 40:
                latency.append(4)
                bw.append(15)
            elif 40 <= i < 44:
                latency.append(2)
                bw.append(20)
            elif 44 <= i < 48:
                latency.append(1)
                bw.append(25)
            else:
                latency.append(1)
                bw.append(25)

    return latency, bw