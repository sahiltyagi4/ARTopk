import os
import math
import socket
import logging
import argparse
import numpy as np
from time import perf_counter_ns
from pymoo.optimize import minimize
from pymoo.decomposition.asf import ASF
from pymoo.operators.mutation.pm import PM
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.termination import get_termination
from pymoo.operators.sampling.rnd import FloatRandomSampling

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ReduceOp

from flexible_compression.miscellaneous import helper, models
import flexible_compression.miscellaneous.datapartitioner as dp
from flexible_compression.miscellaneous.modulate_alphabeta import ModulateLatencyBandwidth
from flexible_compression.miscellaneous.compression import MultiObjOptimizeCompression
from flexible_compression.adaptive_moo.model_moo import ModelMultiObjectiveOptimization

mill = 1e6


# Adaptive communication with compression warmup (i.e, compression is gradually increased over steps/epochs w/ warmup)
class MOOCompressionTraining(object):
    def __init__(self, args):
        self.args = args
        self.train_bsz = args.bsz
        self.test_bsz = args.test_bsz
        self.logdir = args.dir
        self.model_name = args.model
        self.rank = args.rank
        self.worldsize = args.world_size
        self.backend = args.backend
        self.init_method = args.init_method
        self.window_size = args.windowsize
        self.alpha = args.alpha

        if self.init_method == 'sharedfile':
            sharedfile = 'file://' + args.shared_file
            dist.init_process_group(backend=self.backend, init_method=sharedfile, rank=self.rank,
                                    world_size=self.worldsize)
        elif self.init_method == 'tcp':
            import datetime
            timeout = datetime.timedelta(seconds=3000 * 60 * 60 * 100)
            tcp_addr = 'tcp://' + str(args.master_addr) + ':' + str(args.master_port)
            dist.init_process_group(backend=self.backend, init_method=tcp_addr, rank=self.rank,
                                    world_size=self.worldsize, timeout=timeout)

        logging.basicConfig(filename=self.logdir + '/g' + str(self.rank) + '/' + self.model_name + '-' + str(self.rank)
                                     + '.log', level=logging.INFO)

        if args.determinism == 0:
            self.determinism = False
        elif args.determinism == 1:
            self.determinism = True

        if args.async_op == 'False':
            self.async_op = False
        elif args.async_op == 'True':
            self.async_op = True

        self.dataset_name = args.dataset
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(self.rank))
        else:
            self.device = torch.device("cpu")

        self.dataset_obj = dp.Dataset(dataset_name=self.dataset_name, args=args)
        self.model_obj = models.get_model(model_name=self.model_name, determinism=self.determinism, args=args)
        self.model = self.model_obj.get_model()
        self.concat_tensor = torch.cat([torch.zeros(size=p.size()).view(-1) for p in self.model.parameters()]).reshape(-1)
        self.concat_size = self.concat_tensor.size()
        self.concat_elmnt = self.concat_tensor.numel()
        del self.concat_tensor
        self.model = self.model.to(self.device)
        self.loss = self.model_obj.get_loss()
        self.opt = self.model_obj.get_optim()
        self.lr_scheduler = self.model_obj.get_lrscheduler()
        self.globalstep = 0

        self.trainloader = self.dataset_obj.get_trainloader()
        self.testloader = self.dataset_obj.get_testloader()
        self.epochs = args.epochs
        self.train_steps = args.trainsteps
        self.test_steps = args.teststeps
        self.epoch_warmup = args.epoch_warmup
        args.total_dataset_size = self.dataset_obj.get_trainsize()
        global_bsz = self.train_bsz * self.worldsize
        self.warmup_steps = int((self.epoch_warmup * args.total_dataset_size) / global_bsz)
        self.compression = MultiObjOptimizeCompression(device=self.device, concat_size=self.concat_size,
                                                       concat_elmnt=self.concat_elmnt)

        self.latency, self.bandwidth = helper.initialize_latency_bandwidth(model_name=self.model_name)
        self.candidateCRs = [0.1, 0.033, 0.011, 0.004, 0.001]
        self.init_cr = self.candidateCRs[0]
        self.warmup_cr = np.flip(np.linspace(self.init_cr, 1.0, self.warmup_steps))
        self.gain_change_threshold = args.gain_change_threshold
        self.collectives = ['artopk_ring', 'artopk_tree', 'all_gather']
        self.model_chkpt = os.path.join(os.path.join(self.logdir, 'g' + str(self.rank)), 'model.pth')
        self.exploration_itrs = args.explore_itrs
        self.cr_gains, self.cr_collectives, self.cr_compress_cost, self.cr_commtimes = {}, {}, {}, {}
        self.deltaGain_smoother = helper.EWMAMeter(windowsize=self.window_size, alpha=self.alpha)
        self.interface = args.interface
        self.max_bw = args.max_bw
        self.modulate_alphabeta = ModulateLatencyBandwidth(interface=self.interface, worker_rank=self.rank,
                                                           world_size=self.worldsize, max_bw=self.max_bw)

        args.hostname = socket.gethostname()
        args.opt_name = self.opt.__class__.__name__
        logging.info(f'model arguments are {args}')

    def optimizeMultiObjectives(self, comm_times, compress_gains, commtime_crs, gains_crs, curr_epoch, alpha, bw):
        # 3 metrics: compression time, comm. time and compression gain. using latter 2 as first is constant w.r.t. CR
        logging.info(f'optimizing via pymoo for comm_times {comm_times} and gains {compress_gains} latency {alpha} '
                     f'bw {bw} for candidate_crs {self.candidateCRs} commtime_crs {commtime_crs} gain_crs {gains_crs} '
                     f'at step {self.globalstep} epoch {curr_epoch}')
        problem = ModelMultiObjectiveOptimization(candidateCRs=self.candidateCRs, comm_times=comm_times,
                                                  compress_gains=compress_gains)
        algorithm = NSGA2(pop_size=40, n_offsprings=10, sampling=FloatRandomSampling(), crossover=SBX(prob=0.9, eta=15),
                          mutation=PM(eta=20), eliminate_duplicates=True)
        termination = get_termination("n_gen", 40)
        res = minimize(problem, algorithm, termination, seed=1, save_history=True, verbose=False)
        X = res.X
        F = res.F
        approx_ideal = F.min(axis=0)
        approx_nadir = F.max(axis=0)
        nF = (F - approx_ideal) / (approx_nadir - approx_ideal)
        # assigns weightage to communication time and compression gain: here 0.4 to gain and 0.6 to time.
        weights = np.array([0.6, 0.4])
        decomp = ASF()
        c = decomp.do(nF, 1 / weights).argmin()
        optimal_cr = X[c]
        optimal_commtime, optimal_compressgain = F[c][0], F[c][1]

        return optimal_cr, optimal_commtime, optimal_compressgain

    def getEpochAlphaBetaCost(self, curr_epoch, curr_cr):
        bw = self.bandwidth[curr_epoch] * 1.25 * 1e8
        alpha = self.latency[curr_epoch]
        comm_times = self.getLatencyBWCommTime(curr_cr=curr_cr, alpha=alpha, bw=bw)
        return comm_times

    # infer cost of different models based on their alpha-beta communication times
    def getLatencyBWCommTime(self, curr_cr, alpha, bw):
        # model-size of different DNNs
        if self.model_name == 'resnet18':
            m = 11689512
        elif self.model_name == 'resnet50':
            m = 25557032
        elif self.model_name == 'alexnet':
            m = 61100840
        elif self.model_name == 'vision_transformer':
            m = 304326632

        def artopk_ring(alpha, bw, m, cr):
            bcast = alpha * math.log(self.worldsize, 2) + (math.log(self.worldsize, 2) * m * 4 * cr) / bw
            latency_cost = 2 * (self.worldsize - 1) * alpha
            bw_cost = (2 * (self.worldsize - 1) * m * 4 * cr) / (self.worldsize * bw)
            total_cost = bcast + latency_cost + bw_cost
            return total_cost * 1e3

        def artopk_tree(alpha, bw, m, cr):
            bcast = alpha * math.log(self.worldsize, 2) + (math.log(self.worldsize, 2) * m * 4 * cr) / bw
            latency_cost = 2 * alpha * math.log(self.worldsize, 2)
            bw_cost = (2 * math.log(self.worldsize, 2) * m * 4 * cr) / bw
            total_cost = bcast + latency_cost + bw_cost
            return total_cost * 1e3

        def all_gather(alpha, bw, m, cr):
            latency_cost = alpha * math.log(self.worldsize, 2)
            bw_cost = ((self.worldsize - 1) * m * 4 * cr * 2) / bw
            total_cost = latency_cost + bw_cost
            return total_cost * 1e3

        # collectives used in this order = ['artopk_ring', 'artopk_tree', 'all_gather']
        comm_times = [artopk_ring(alpha=alpha * 1e-3, bw=bw, m=m, cr=curr_cr), artopk_tree(alpha=alpha * 1e-3, bw=bw, m=m, cr=curr_cr),
                      all_gather(alpha=alpha * 1e-3, bw=bw, m=m, cr=curr_cr)]

        return comm_times

    def accuracy(self, output, target, topk=(1,)):
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

    def test_accuracy(self, curr_epoch, prefix='STEP'):
        if self.globalstep >= self.test_steps and self.globalstep % self.test_steps == 0:
            self.model.eval()
            with torch.no_grad():
                top1acc, top5acc, top10acc = helper.AverageMeter(), helper.AverageMeter(), helper.AverageMeter()
                test_loss = helper.AverageMeter()
                for input, label in self.testloader:
                    input, label = input.to(self.device), label.to(self.device)
                    output = self.model(input)
                    loss = self.loss(output, label)
                    topaccs = self.accuracy(output=output, target=label, topk=(1, 5, 10))
                    top1acc.update(topaccs[0], input.size(0))
                    top5acc.update(topaccs[1], input.size(0))
                    top10acc.update(topaccs[2], input.size(0))
                    test_loss.update(loss.item(), input.size(0))

                logging.info(
                    f'{prefix} VALIDATION METRICS step {self.globalstep} epoch {curr_epoch} testlossval {test_loss.val} '
                    f'testlossavg {test_loss.avg} top1val {top1acc.val.cpu().numpy().item()} '
                    f'top1avg {top1acc.avg.cpu().numpy().item()} top5val {top5acc.val.cpu().numpy().item()} '
                    f'top5avg {top5acc.avg.cpu().numpy().item()} top10val {top10acc.val.cpu().numpy().item()} '
                    f'top10avg {top10acc.avg.cpu().numpy().item()}')

            self.model.train()

    def train_accuracy(self, input, label, output, loss, epoch):
        if self.globalstep >= self.train_steps and self.globalstep % self.train_steps == 0:
            with torch.no_grad():
                trainaccs = self.accuracy(output=output, target=label, topk=(1, 5, 10))
                self.top1accs.update(trainaccs[0], input.size(0))
                self.top5accs.update(trainaccs[1], input.size(0))
                self.top10accs.update(trainaccs[2], input.size(0))
                self.train_loss.update(loss.item(), input.size(0))

                logging.info(
                    f'TRAINING METRICS step {self.globalstep} epoch {epoch} trainlossval {self.train_loss.val} '
                    f'trainlossavg {self.train_loss.avg} top1val {self.top1accs.val.cpu().numpy().item()} '
                    f'top1avg {self.top1accs.avg.cpu().numpy().item()} top5val {self.top5accs.val.cpu().numpy().item()} '
                    f'top5avg {self.top5accs.avg.cpu().numpy().item()} top10val {self.top10accs.val.cpu().numpy().item()} '
                    f'top10avg {self.top10accs.avg.cpu().numpy().item()}')

    def updateMOOParameters(self, curr_epoch, curr_step):
        # when done, delete the checkpoint and broadcast parameters from rank 0 worker
        torch.save(self.model.state_dict(), self.model_chkpt)
        for test_cr in self.candidateCRs:
            self.model.load_state_dict(torch.load(self.model_chkpt))
            self.model.train()
            compress_timelist = []
            compression = MultiObjOptimizeCompression(device=self.device, concat_size=self.concat_size,
                                                      concat_elmnt=self.concat_elmnt)
            comm_times = self.getEpochAlphaBetaCost(curr_epoch=curr_epoch, curr_cr=test_cr)
            comm_collective = self.collectives[comm_times.index(min(comm_times))]
            gain_smoothing = helper.EWMAMeter(windowsize=5, alpha=0.01)
            eval_strt = perf_counter_ns()
            for j in range(self.exploration_itrs):
                input, label = next(iter(self.testloader))
                input, label = input.to(self.device), label.to(self.device)
                output = self.model(input)
                loss = self.loss(output, label)
                loss.backward()

                grads = torch.cat([p.grad.data.view(-1) for p in self.model.parameters()]).reshape(-1)
                grads_size = grads.size()

                if comm_collective == 'all_gather':
                    begin = perf_counter_ns()
                    cmprss_vals, cmprss_ixs = compression.ag_compress(tensor=grads, cr=test_cr)
                    compress_time = (perf_counter_ns() - begin) / mill
                    compress_timelist.append(compress_time)

                    compressgain_tnsr = torch.FloatTensor([compression.compress_gain()])
                    compressgain_tnsr = compressgain_tnsr.to(self.device)
                    dist.all_reduce(tensor=compressgain_tnsr, op=ReduceOp.SUM, async_op=self.async_op)
                    compressgain_tnsr = torch.div(compressgain_tnsr, self.worldsize)
                    cmprss_gain = compressgain_tnsr.item()
                    gain_smoothing.smoothendata(cmprss_gain)

                    t_s = cmprss_vals.numel()
                    t_list, ix_list = [], []
                    for _ in range(self.worldsize):
                        t_list.append(torch.zeros(size=(t_s,), dtype=torch.float32, device=self.device))
                        ix_list.append(torch.zeros(size=(t_s,), dtype=torch.long, device=self.device))

                    dist.all_gather(t_list, cmprss_vals, async_op=self.async_op)
                    dist.all_gather(ix_list, cmprss_ixs, async_op=self.async_op)
                    grads = torch.zeros(size=grads_size, device=self.device)
                    for ix in range(len(t_list)):
                        grads.data[ix_list[ix]] += t_list[ix]

                    grads = torch.div(grads, self.worldsize)

                elif comm_collective == 'artopk_ring' or comm_collective == 'artopk_tree':
                    # implements STAR-Topk compression
                    bcast_rank = self.globalstep % self.worldsize
                    if bcast_rank != self.rank:
                        ix_len = torch.LongTensor([0])
                        ix_len = ix_len.to(self.device)
                        cmptime_tensor = torch.FloatTensor([0])
                        cmptime_tensor = cmptime_tensor.to(self.device)
                        compressgain_tnsr = torch.FloatTensor([0])
                        compressgain_tnsr = compressgain_tnsr.to(self.device)
                    elif bcast_rank == self.rank:
                        begin = perf_counter_ns()
                        cmprss_vals, cmprss_ixs = compression.artopk_compress(tensor=grads, cr=test_cr)
                        compress_time = (perf_counter_ns() - begin) / mill
                        ix_len = torch.LongTensor([cmprss_ixs.size()[0]])
                        ix_len = ix_len.to(self.device)
                        cmptime_tensor = torch.FloatTensor([compress_time])
                        cmptime_tensor = cmptime_tensor.to(self.device)
                        compressgain_tnsr = torch.FloatTensor([compression.compress_gain()])
                        compressgain_tnsr = compressgain_tnsr.to(self.device)

                    dist.broadcast(tensor=ix_len, src=bcast_rank, async_op=self.async_op)
                    dist.broadcast(tensor=cmptime_tensor, src=bcast_rank, async_op=self.async_op)
                    dist.broadcast(tensor=compressgain_tnsr, src=bcast_rank, async_op=self.async_op)
                    compress_timelist.append(cmptime_tensor.item())
                    cmprss_gain = compressgain_tnsr.item()
                    gain_smoothing.smoothendata(cmprss_gain)
                    if bcast_rank != self.rank:
                        cmprss_ixs = torch.zeros(size=[ix_len.item()], dtype=torch.int64)

                    cmprss_ixs = cmprss_ixs.to(self.device)
                    dist.broadcast(tensor=cmprss_ixs, src=bcast_rank, async_op=self.async_op)
                    if bcast_rank != self.rank:
                        cmprss_vals = compression.fetch_artopk_ixs(tensor=grads, ix_tensor=cmprss_ixs)

                    dist.all_reduce(tensor=cmprss_vals, op=ReduceOp.SUM, async_op=self.async_op)
                    cmprss_vals = torch.div(cmprss_vals, self.worldsize)
                    grads = torch.zeros(size=grads_size, device=self.device)
                    grads.data[cmprss_ixs] = cmprss_vals

                for p in self.model.parameters():
                    p_shape = list(p.size())
                    p_len = p.numel()
                    p.grad.data = grads[0:p_len].reshape(p_shape)
                    grads = grads[p_len:]

                self.opt.step()
                self.opt.zero_grad()

            self.cr_gains[test_cr] = gain_smoothing.smooth_val()
            self.cr_collectives[test_cr] = comm_collective
            self.cr_compress_cost[test_cr] = np.mean(compress_timelist)
            self.cr_commtimes[test_cr] = min(comm_times)

        self.model.load_state_dict(torch.load(self.model_chkpt))
        os.remove(self.model_chkpt)
        eval_time = (perf_counter_ns() - eval_strt) / mill
        logging.info(f'executed updateMOOParameters at epoch {curr_epoch} step {curr_step} eval_time {eval_time} ms')

    def updateCommunicationCost(self, latency, bw):
        beta_bw = bw * 1.25 * 1e8
        for test_cr in self.candidateCRs:
            comm_times = self.getLatencyBWCommTime(curr_cr=test_cr, alpha=latency, bw=beta_bw)
            comm_collective = self.collectives[comm_times.index(min(comm_times))]
            self.cr_collectives[test_cr] = comm_collective
            self.cr_commtimes[test_cr] = min(comm_times)

    def start_training(self):
        one_time_flag = True
        previous_gain = 0.
        last_latency, last_bw = self.latency[0], self.bandwidth[0]
        for _, param in self.model.named_parameters():
            if not param.requires_grad: continue
            dist.broadcast(tensor=param.data, src=0, async_op=self.async_op)

        for epoch in range(0, self.epochs):
            self.top1accs, self.top5accs, self.top10accs = helper.AverageMeter(), helper.AverageMeter(), helper.AverageMeter()
            self.train_loss = helper.AverageMeter()
            curr_latency, curr_bw = self.latency[epoch], self.bandwidth[epoch]
            self.modulate_alphabeta.adjustAlphaBeta(latency=curr_latency, bandwidth=curr_bw)
            for record in self.trainloader:
                input, label = record
                input, label = input.to(self.device), label.to(self.device)
                begin = perf_counter_ns()
                output = self.model(input)
                loss = self.loss(output, label)
                loss.backward()
                compute_time = (perf_counter_ns() - begin) / mill
                grads = torch.cat([p.grad.data.view(-1) for p in self.model.parameters()]).reshape(-1)
                grads_size = grads.size()

                if self.globalstep < self.warmup_steps:
                    cr = self.warmup_cr[self.globalstep]
                    comm_times = self.getEpochAlphaBetaCost(curr_epoch=epoch, curr_cr=cr)
                    curr_comm_time = min(comm_times)
                    comm_collective = self.collectives[comm_times.index(curr_comm_time)]
                    gain_change = None

                if comm_collective == 'all_gather':
                    begin = perf_counter_ns()
                    cmprss_vals, cmprss_ixs = self.compression.ag_compress(tensor=grads, cr=cr)
                    compress_time = (perf_counter_ns() - begin) / mill
                    gain_tnsr = torch.FloatTensor([self.compression.compress_gain()])
                    gain_tnsr = gain_tnsr.to(self.device)
                    dist.all_reduce(tensor=gain_tnsr, op=ReduceOp.SUM, async_op=self.async_op)
                    gain_tnsr = torch.div(gain_tnsr, self.worldsize)
                    compression_gain = gain_tnsr.item()
                    t_s = cmprss_vals.numel()
                    t_list, ix_list = [], []
                    for _ in range(self.worldsize):
                        t_list.append(torch.zeros(size=(t_s,), dtype=torch.float32, device=self.device))
                        ix_list.append(torch.zeros(size=(t_s,), dtype=torch.long, device=self.device))

                    dist.all_gather(t_list, cmprss_vals, async_op=self.async_op)
                    dist.all_gather(ix_list, cmprss_ixs, async_op=self.async_op)
                    grads = torch.zeros(size=grads_size, device=self.device)
                    for ix in range(len(t_list)):
                        grads.data[ix_list[ix]] += t_list[ix]

                    grads = torch.div(grads, self.worldsize)

                elif comm_collective == 'artopk_ring' or comm_collective == 'artopk_tree':
                    bcast_rank = self.globalstep % self.worldsize
                    if bcast_rank != self.rank:
                        ix_len = torch.LongTensor([0])
                        ix_len = ix_len.to(self.device)
                        bcast_compgain = torch.FloatTensor([0])
                        bcast_compgain = bcast_compgain.to(self.device)
                        cmprsstime_tnsr = torch.FloatTensor([0])
                        cmprsstime_tnsr = cmprsstime_tnsr.to(self.device)
                    elif bcast_rank == self.rank:
                        begin = perf_counter_ns()
                        cmprss_vals, cmprss_ixs = self.compression.artopk_compress(tensor=grads, cr=cr)
                        compress_time = (perf_counter_ns() - begin) / mill
                        ix_len = torch.LongTensor([cmprss_ixs.size()[0]])
                        ix_len = ix_len.to(self.device)
                        bcast_compgain = torch.FloatTensor([self.compression.compress_gain()])
                        bcast_compgain = bcast_compgain.to(self.device)
                        cmprsstime_tnsr = torch.FloatTensor([compress_time])
                        cmprsstime_tnsr = cmprsstime_tnsr.to(self.device)

                    dist.broadcast(tensor=ix_len, src=bcast_rank, async_op=self.async_op)
                    dist.broadcast(tensor=bcast_compgain, src=bcast_rank, async_op=self.async_op)
                    dist.broadcast(tensor=cmprsstime_tnsr, src=bcast_rank, async_op=self.async_op)
                    compression_gain = bcast_compgain.item()
                    compress_time = cmprsstime_tnsr.item()
                    if bcast_rank != self.rank:
                        cmprss_ixs = torch.zeros(size=[ix_len.item()], dtype=torch.int64)
                        cmprss_ixs = cmprss_ixs.to(self.device)

                    dist.broadcast(tensor=cmprss_ixs, src=bcast_rank, async_op=self.async_op)
                    if bcast_rank != self.rank:
                        cmprss_vals = self.compression.fetch_artopk_ixs(tensor=grads, ix_tensor=cmprss_ixs)

                    dist.all_reduce(tensor=cmprss_vals, op=ReduceOp.SUM, async_op=self.async_op)
                    cmprss_vals = torch.div(cmprss_vals, self.worldsize)
                    grads = torch.zeros(size=grads_size, device=self.device)
                    grads.data[cmprss_ixs] = cmprss_vals
                    torch.cuda.synchronize(self.device)

                for p in self.model.parameters():
                    p_shape = list(p.size())
                    p_len = p.numel()
                    p.grad.data = grads[0:p_len].reshape(p_shape)
                    grads = grads[p_len:]

                if self.globalstep >= self.warmup_steps:
                    # run once after warm-up phase
                    if one_time_flag:
                        self.updateMOOParameters(curr_epoch=epoch, curr_step=self.globalstep)
                        one_time_flag = False

                    comm_times = self.getEpochAlphaBetaCost(curr_epoch=epoch, curr_cr=cr)
                    curr_comm_time = min(comm_times)

                    gain_delta = abs((compression_gain - previous_gain) / previous_gain)
                    self.deltaGain_smoother.smoothendata(gain_delta)
                    gain_change = self.deltaGain_smoother.smooth_val()
                    if gain_change >= self.gain_change_threshold:
                        logging.info(f'found gain_change {gain_change} at epoch {epoch} step {self.globalstep}')
                        self.updateMOOParameters(curr_epoch=epoch, curr_step=self.globalstep)

                    if curr_latency != last_latency or curr_bw != last_bw:
                        self.updateCommunicationCost(latency=curr_latency, bw=curr_bw)
                        # returns optimal CR and its estimated communication time and compression gain
                        opt_cr, opt_commtime, opt_gain = self.optimizeMultiObjectives(comm_times=list(self.cr_commtimes.values()),
                                                                                      compress_gains=list(self.cr_gains.values()),
                                                                                      commtime_crs=list(self.cr_commtimes.keys()),
                                                                                      gains_crs=list(self.cr_gains.keys()),
                                                                                      curr_epoch=epoch, alpha=curr_latency,
                                                                                      bw=curr_bw)
                        last_latency, last_bw = curr_latency, curr_bw
                        cr = opt_cr
                        beta_bw = curr_bw * 1.25 * 1e8
                        est_comm_times = self.getLatencyBWCommTime(curr_cr=cr, alpha=curr_latency, bw=beta_bw)
                        est_comm_time = min(est_comm_times)
                        curr_comm_time = est_comm_time
                        comm_collective = self.collectives[est_comm_times.index(est_comm_time)]
                        logging.info(
                            f'multi-objective optimization found cr {opt_cr} for commtime {opt_commtime} '
                            f'all_commtimes {est_comm_times} gain {opt_gain} and est_commtime '
                            f'{est_comm_time} collective {comm_collective} at step {self.globalstep} epoch {epoch} for '
                            f'latency {curr_latency} and bw {curr_bw}')

                previous_gain = compression_gain
                sched_lr = self.lr_scheduler.get_last_lr()[0]
                self.globalstep += 1
                self.opt.step()
                self.opt.zero_grad()
                logging.info(
                    f'moo_compression WARMUP epoch {epoch} step {self.globalstep} compute_time {compute_time} ms '
                    f'compress_time {compress_time} ms comm_collective {comm_collective} '
                    f'compress_gain {compression_gain} lr {sched_lr} cr {cr} gain_change {gain_change} '
                    f'curr_alpha {curr_latency} last_alpha {last_latency} curr_bw {curr_bw} last_bw {last_bw} '
                    f'curr_comm_time {curr_comm_time}')
                self.train_accuracy(input=input, label=label, output=output, loss=loss, epoch=epoch)
                self.test_accuracy(curr_epoch=epoch, prefix='STEP')

            self.lr_scheduler.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1234, help='seed value for replication')
    parser.add_argument('--dir', type=str, default='/')
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--backend', type=str, default='gloo')
    parser.add_argument('--interface', type=str, default='eth0')
    parser.add_argument('--max-bw', type=int, default=25, help='maximum bandwidth available/supported by the interface')
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')
    parser.add_argument('--master-port', type=str, default='28564')
    parser.add_argument('--interface', type=str, default='eth0')
    parser.add_argument('--init-method', type=str, default='sharedfile')
    parser.add_argument('--shared-file', type=str, default='/')
    parser.add_argument('--test-bsz', type=int, default=32)
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--determinism', type=int, default=0)
    parser.add_argument('--trainsteps', type=int, default=100)
    parser.add_argument('--teststeps', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--train-dir', type=str, default='/train_dir')
    parser.add_argument('--test-dir', type=str, default='/val_dir')
    parser.add_argument('--windowsize', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--gain-change-threshold', type=float, default=0.1)
    parser.add_argument('--explore-itrs', type=int, default=10)
    parser.add_argument('--async-op', type=str, default='False')
    parser.add_argument('--epoch-warmup', type=int, default=5, help='# of epochs to run compression-warmup')

    args = parser.parse_args()

    if args.backend == 'gloo':
        os.environ['GLOO_SOCKET_IFNAME'] = args.interface
    elif args.backend == 'nccl':
        # os.environ['NCCL_IB_DISABLE'] = '1'
        os.environ['NCCL_SOCKET_IFNAME'] = args.interface

    MOOCompressionTraining(args).start_training()