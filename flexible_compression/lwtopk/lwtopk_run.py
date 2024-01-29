import logging
import socket
import argparse
from time import perf_counter_ns

import torch
import torch.distributed as dist

from flexible_compression.miscellaneous import helper, models
from flexible_compression.miscellaneous.compression import LWTopKCompression
import flexible_compression.miscellaneous.datapartitioner as dp

class LWTopKModelTraining(object):
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
        self.dataset_name = args.dataset
        if args.determinism == 0:
            self.determinism = False
        elif args.determinism == 1:
            self.determinism = True

        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(self.rank))
        else:
            self.device = torch.device("cpu")

        self.dataset_obj = dp.Dataset(dataset_name=self.dataset_name, args=args)
        self.model_obj = models.get_model(model_name=self.model_name, determinism=self.determinism, args=args)

        param_names = []
        param_shapes, param_lengths = {}, {}
        self.param_shapes, self.param_lengths = [], []
        for name, param in self.model_obj.get_model().named_parameters():
            if param.requires_grad:
                param_names.append(name)
                param_shapes[name] = param.size()
                param_lengths[name] = param.numel()
                self.param_shapes.append(list(param.size()))
                self.param_lengths.append(param_lengths[name])

        self.model = self.model_obj.get_model().to(self.device)
        self.loss = self.model_obj.get_loss()
        self.opt = self.model_obj.get_optim()
        self.lr_scheduler = self.model_obj.get_lrscheduler()
        self.globalstep = 0
        self.windowsize = args.windowsize
        self.smoothing = args.smoothing
        self.model_stats = helper.ModelStatistics(param_names=param_names, param_shapes=param_shapes,
                                                  param_lengths=param_lengths, windowsize=self.windowsize,
                                                  smoothing=self.smoothing)
        del param_names, param_shapes, param_lengths
        self.trainloader = self.dataset_obj.get_trainloader()
        self.testloader = self.dataset_obj.get_testloader()
        self.epochs = args.epochs
        self.train_steps = args.trainsteps
        self.test_steps = args.teststeps

        # compression ratio
        self.cr = args.cr
        self.compression = LWTopKCompression(device=self.device, cr=self.cr)
        if args.async_op == 'False':
            self.async_op = False
        elif args.async_op == 'True':
            self.async_op = True

        self.ef_ewma = helper.EWMAMeter(windowsize=self.windowsize, alpha=self.smoothing)
        self.cf_ewma = helper.EWMAMeter(windowsize=self.windowsize, alpha=self.smoothing)
        args.hostname = socket.gethostname()
        args.opt_name = self.opt.__class__.__name__
        logging.info(f'model arguments are {args}')

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
                    topaccs = self.model_stats.compute_imgaccuracy(output=output, target=label, topk=(1, 5, 10))
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
                trainaccs = self.model_stats.compute_imgaccuracy(output=output, target=label, topk=(1, 5, 10))
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

    def launch_training(self):
        mill = 1e6
        for epoch in range(0, self.epochs):
            self.top1accs, self.top5accs, self.top10accs = helper.AverageMeter(), helper.AverageMeter(), helper.AverageMeter()
            self.train_loss = helper.AverageMeter()
            self.top100accs, self.top1000accs = helper.AverageMeter(), helper.AverageMeter()
            for record in self.trainloader:
                input, label = record
                input, label = input.to(self.device), label.to(self.device)
                self.globalstep += 1
                begin = perf_counter_ns()
                output = self.model(input)
                loss = self.loss(output, label)
                loss.backward()
                compute_time = (perf_counter_ns() - begin) / mill

                begin = perf_counter_ns()
                with torch.no_grad():
                    lw_vals, lw_ixs = [], []
                    for name, p in self.model.named_parameters():
                        tnsr_compress = self.compression.compress(p.grad.data, name)
                        lw_vals.append(tnsr_compress[0])
                        lw_ixs.append(tnsr_compress[1])

                compress_time = (perf_counter_ns() - begin) / mill

                begin = perf_counter_ns()
                reduced_grads = []
                for i in range(len(lw_vals)):
                    # compressed gradient
                    c_grad = lw_vals[i]
                    c_ix = lw_ixs[i]
                    t_sizes = [torch.LongTensor([0]) for _ in range(self.worldsize)]  # fetch tensor size from all ranks
                    t_s = c_grad.numel()
                    dist.all_gather(t_sizes, torch.LongTensor([t_s]), async_op=self.async_op)

                    # tensor list and index list
                    t_list, ix_list = [], []
                    # collect size from tensor sizes of Top-K
                    sz_list = [int(sz.item()) for sz in t_sizes]
                    max_sz = max(sz_list)
                    if max_sz > 0:
                        for _ in sz_list:
                            t_list.append(torch.zeros(size=(max_sz,), dtype=torch.float32, device=self.device))
                            ix_list.append(torch.zeros(size=(max_sz,), dtype=torch.long, device=self.device))
                        if t_s != max_sz:
                            g_padding = torch.zeros(size=(max_sz - t_s,), dtype=torch.float32, device=self.device)
                            ix_padding = torch.zeros(size=(max_sz - t_s,), dtype=torch.long, device=self.device)
                            c_grad = torch.cat((c_grad, g_padding), dim=0)
                            c_ix = torch.cat((c_ix, ix_padding), dim=0)

                        dist.all_gather(t_list, c_grad, async_op=self.async_op)
                        dist.all_gather(ix_list, c_ix, async_op=self.async_op)
                        p_shape = self.param_shapes[i]
                        tnsr = torch.zeros(size=p_shape, device=self.device).view(-1)
                        for ix in range(len(t_list)):
                            tnsr.data[ix_list[ix]] += t_list[ix]

                        tnsr /= self.worldsize
                        tnsr = tnsr.reshape(p_shape)
                        reduced_grads.append(tnsr)

                    else:
                        reduced_grads.append(None)

                sync_time = (perf_counter_ns() - begin) / mill
                for p, g in zip(self.model.parameters(), reduced_grads):
                    p.grad.data = g

                sched_lr = self.lr_scheduler.get_last_lr()[0]
                self.opt.step()
                self.opt.zero_grad()

                logging.info(f'imglog tstep {self.globalstep} epoch {epoch} compute_time {compute_time} ms lr {sched_lr} '
                             f'sync_time {sync_time} ms compress_time ms {compress_time} ms cr {self.cr}')
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
    parser.add_argument('--master-addr', type=str, default='127.0.0.1')
    parser.add_argument('--master-port', type=str, default='28564')
    parser.add_argument('--interface', type=str, default='NULL')
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
    parser.add_argument('--train-dir', type=str, default='/')
    parser.add_argument('--test-dir', type=str, default='/')
    parser.add_argument('--windowsize', type=int, default=25)
    parser.add_argument('--smoothing', type=float, default=0.001)
    parser.add_argument('--compress-method', type=str, default='None')
    parser.add_argument('--async-op', type=str, default='False')
    parser.add_argument('--cr', type=float, default=0.1)

    args = parser.parse_args()

    LWTopKModelTraining(args).launch_training()