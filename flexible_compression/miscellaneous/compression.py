import math
import random

import torch

from flexible_compression.miscellaneous import Compressor
from flexible_compression.miscellaneous.residual import ResidualMemory


def sparsify(tensor, compress_ratio):
    tensor = tensor.flatten()
    k = max(1, int(tensor.numel() * compress_ratio))
    _, indices = torch.topk(tensor.abs(), k, sorted=False,)
    values = torch.gather(tensor, 0, indices)
    return values, indices
    # k = max(1, int(tensor.numel() * compress_ratio))
    # values, indexes = tensor.abs().sort(descending=True)
    # return values[:k], indexes[:k]


def desparsify(tensors, numel, device):
    values, indices = tensors
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=device)
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed


def deparse(values, indices, numel, device):
    tensor_decompressed = torch.zeros(numel, dtype=values.dtype, device=device)
    tensor_decompressed.scatter_(0, indices, values)
    return tensor_decompressed


class LWTopKCompression(Compressor):

    def __init__(self, device, cr):
        super().__init__()
        self.residual = ResidualMemory()
        self.device = device
        self.cr = cr

    def compress(self, tensor, name):
        tensor = tensor.to(self.device)

        tensor = self.residual.compensate(tensor, name)
        numel = tensor.numel()
        shape = tensor.size()
        tensors = sparsify(tensor, self.cr)
        ctx = numel, shape
        self.residual.update(tensor, name, self, tensors, ctx)
        return tensors

    def decompress(self, tensors, ctx):
        """Decompress by filling empty slots with zeros and reshape back using the original shape"""
        numel, shape = ctx
        tensor_decompressed = desparsify(tensors, numel, self.device)
        return tensor_decompressed.view(shape)


class MSTopKCompression(object):

    def __init__(self, device, cr, concat_size, concat_elmnt, N):
        self.N = N
        self.device = device
        self.cr = cr
        self.concat_size = concat_size
        self.concat_elmnt = concat_elmnt
        self.residual = torch.empty(size=self.concat_size, device=self.device)
        #  number of elements in compressed tensor
        self.k = math.floor(self.cr * self.concat_elmnt)

    def deparse(self, vals, ixs):
        tensor_decompressed = torch.zeros(self.concat_elmnt, dtype=vals.dtype, device=self.device)
        tensor_decompressed.scatter_(0, ixs, vals)
        return tensor_decompressed

    def msampling_compress(self, tensor):
        tensor = tensor.to(self.device)
        tensor = tensor + self.residual

        a = torch.abs(tensor)
        a_hat = torch.mean(a)
        u = torch.max(a)
        l, r = 0, 1
        k1, k2 = 0, tensor.numel()
        thres1, thres2 = 0, 0
        for i in range(self.N):
            ratio = l + (r - l) / 2
            thres = a_hat + ratio * (u - a_hat)
            nnz = torch.count_nonzero(a >= thres)
            if nnz <= self.k:
                r = ratio
                if nnz > k1:
                    k1 = nnz
                    thres1 = thres
            elif nnz > self.k:
                l = ratio
                if nnz < k2:
                    k2 = nnz
                    thres2 = thres

        l1 = torch.nonzero(a >= thres1, as_tuple=True)[0]
        l2 = torch.nonzero((a < thres1) & (a >= thres2), as_tuple=True)[0]
        rand = random.randint(0, len(l2) - (self.k - k1) + 1)
        cmprss_ix = torch.concat((l1, l2[rand : rand + self.k - k1]))
        cmprss_vals = tensor[cmprss_ix]

        tensor_decompressed = self.deparse(vals=cmprss_vals, ixs=cmprss_ix)
        tensor_decompressed = tensor_decompressed.view(self.concat_size)
        self.residual = tensor - tensor_decompressed
        return cmprss_vals, cmprss_ix


class ARConcatTopKCompression(object):

    def __init__(self, device, cr, concat_size, concat_elmnt):
        self.device = device
        self.cr = cr
        self.concat_size = concat_size
        self.concat_elmnt = concat_elmnt
        self.residual = torch.empty(size=self.concat_size, device=self.device)

    def allreduce_topk(self, tensor):
        tensor = tensor.to(self.device)
        tensor = tensor + self.residual

        tensors = sparsify(tensor, self.cr)
        tensor_decompressed = desparsify(tensors, self.concat_elmnt, self.device)
        tensor_decompressed = tensor_decompressed.view(self.concat_size)
        self.residual = tensor - tensor_decompressed
        return tensors

    def fetch_from_indices(self, tensor, ix_tensor):
        tensor = tensor.to(self.device)
        tensor = tensor + self.residual

        values = torch.gather(tensor, 0, ix_tensor)
        tensor_decompressed = torch.zeros(self.concat_elmnt, dtype=values.dtype, device=self.device)
        tensor_decompressed.scatter_(0, ix_tensor, values)
        tensor_decompressed = tensor_decompressed.view(self.concat_size)
        self.residual = tensor - tensor_decompressed
        return values


class ARVarianceTopKCompress(object):

    def __init__(self, device, concat_size, concat_elmnt):
        self.device = device
        self.concat_size = concat_size
        self.concat_elmnt = concat_elmnt
        self.residual = torch.empty(size=self.concat_size, device=self.device)

    def ar_variance_compress(self, tensor, cr):
        tensor = tensor.to(self.device)
        tensor = tensor + self.residual

        tensors = sparsify(tensor, cr)
        tensor_decompressed = desparsify(tensors, self.concat_elmnt, self.device)
        tensor_decompressed = tensor_decompressed.view(self.concat_size)
        self.residual = tensor - tensor_decompressed
        return tensors

    def compress_from_indices(self, tensor, ix_tensor):
        tensor = tensor.to(self.device)
        tensor = tensor + self.residual

        values = torch.gather(tensor, 0, ix_tensor)
        tensor_decompressed = torch.zeros(self.concat_elmnt, dtype=values.dtype, device=self.device)
        tensor_decompressed.scatter_(0, ix_tensor, values)
        tensor_decompressed = tensor_decompressed.view(self.concat_size)
        self.residual = tensor - tensor_decompressed
        return values


class MultiObjOptimizeCompression(object):

    def __init__(self, device, concat_size, concat_elmnt):
        self.device = device
        self.concat_size = concat_size
        self.concat_elmnt = concat_elmnt
        self.residual = torch.empty(size=self.concat_size, device=self.device)
        self.ef_grad, self.cf_grad = 0., 0.

    def artopk_compress(self, tensor, cr):
        tensor = tensor.to(self.device)
        tensor = tensor + self.residual
        self.ef_grad = torch.norm(tensor.contiguous().view(-1))

        tensors = sparsify(tensor, cr)
        self.cf_grad = torch.norm(tensors[0].contiguous().view(-1))
        tensor_decompressed = desparsify(tensors, self.concat_elmnt, self.device)
        tensor_decompressed = tensor_decompressed.view(self.concat_size)
        self.residual = tensor - tensor_decompressed

        return tensors

    def fetch_artopk_ixs(self, tensor, ix_tensor):
        tensor = tensor.to(self.device)
        tensor = tensor + self.residual

        values = torch.gather(tensor, 0, ix_tensor)
        tensor_decompressed = torch.zeros(self.concat_elmnt, dtype=values.dtype, device=self.device)
        tensor_decompressed.scatter_(0, ix_tensor, values)
        tensor_decompressed = tensor_decompressed.view(self.concat_size)
        self.residual = tensor - tensor_decompressed

        return values

    def ag_compress(self, tensor, cr):
        tensor = tensor.to(self.device)
        tensor = tensor + self.residual
        self.ef_grad = torch.norm(tensor.contiguous().view(-1))

        values, indices = sparsify(tensor, cr)
        self.cf_grad = torch.norm(values.contiguous().view(-1))
        tensor_decompressed = deparse(values, indices, self.concat_elmnt, self.device)
        tensor_decompressed = tensor_decompressed.view(self.concat_size)
        self.residual = tensor - tensor_decompressed

        return values, indices

    def compress_gain(self):

        return self.cf_grad / self.ef_grad