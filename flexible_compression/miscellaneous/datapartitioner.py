import random
import os
import math
import shutil
import numpy as np

import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import flexible_compression.miscellaneous.helper as helper


class Dataset(object):
    def __init__(self, dataset_name, args):
        if dataset_name == 'cifar10':
            self.trainloader, self.train_size = cifar10_train(log_dir=args.dir, world_size=args.world_size,
                                                              trainer_rank=args.rank, train_bsz=args.bsz,
                                                              seed=args.seed, num_workers=1)
            self.testloader = cifar10_test(log_dir=args.dir, test_bsz=args.test_bsz, seed=args.seed)
            self.dataset_name = 'CIFAR10'

        elif dataset_name == 'cifar100':
            self.trainloader, self.train_size = cifar100_train(log_dir=args.dir, world_size=args.world_size,
                                                               trainer_rank=args.rank, train_bsz=args.bsz,
                                                               seed=args.seed, num_workers=1)
            self.testloader = cifar100_test(log_dir=args.dir, test_bsz=args.test_bsz, seed=args.seed)
            self.dataset_name = 'CIFAR100'

        elif dataset_name == 'food101':
            self.trainloader, self.train_size = food101_train(log_dir=args.dir, world_size=args.world_size,
                                                              trainer_rank=args.rank, train_bsz=args.bsz,
                                                              seed=args.seed, num_workers=1)
            self.testloader = food101_test(log_dir=args.dir, test_bsz=args.test_bsz, seed=args.seed)

        elif dataset_name == 'caltech256' or dataset_name == 'caltech101':
            self.trainloader, self.train_size = caltech_train(train_dir=args.train_dir, bsz=args.bsz, seed=args.seed,
                                                              world_size=args.world_size, rank=args.rank)
            self.testloader = caltech_test(testdir=args.test_dir, test_bsz=args.test_bsz, seed=args.seed)
            self.dataset_name = 'CalTech'

    def get_trainloader(self):
        return self.trainloader

    def get_testloader(self):
        return self.testloader

    def get_trainsize(self):
        return self.train_size


class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartioner(object):
    def __init__(self, data, world_size):
        self.data = data
        self.partitions = []
        # partition data equally among the trainers
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        random.shuffle(indexes)

        partitions = [1 / (world_size) for _ in range(0, world_size)]
        print(f"partitions are {partitions}")

        for part in partitions:
            part_len = int(part * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


def cifar10_train(log_dir, world_size, trainer_rank, train_bsz, seed, num_workers=1):
    helper.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, 4), normalize])
    trainset = torchvision.datasets.CIFAR10(root=log_dir + 'data', train=True, download=True, transform=transform)
    partition = DataPartioner(trainset, world_size)
    partition = partition.use(trainer_rank)
    trainloader = torch.utils.data.DataLoader(partition, batch_size=train_bsz, shuffle=True,
                                              worker_init_fn=helper.set_seed(seed), generator=g, num_workers=num_workers)
    return trainloader, len(trainset)


def cifar10_test(log_dir, test_bsz, seed):
    helper.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.ToTensor(), normalize])
    testset = torchvision.datasets.CIFAR10(root=log_dir + 'data', train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bsz, shuffle=True, generator=g, num_workers=1)
    return testloader


def cifar100_train(log_dir, world_size, trainer_rank, train_bsz, seed, num_workers=1):
    helper.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),
                                    transforms.Resize(224), normalize])
    trainset = torchvision.datasets.CIFAR100(root=log_dir, train=True, download=True, transform=transform)
    partition = DataPartioner(trainset, world_size)
    partition = partition.use(trainer_rank)
    trainloader = torch.utils.data.DataLoader(partition, batch_size=train_bsz, shuffle=True,
                                              worker_init_fn=helper.set_seed(seed), generator=g,
                                              num_workers=num_workers)
    return trainloader, len(trainset)


def cifar100_test(log_dir, test_bsz, seed):
    helper.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                     std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
    transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), normalize])
    testset = torchvision.datasets.CIFAR100(root=log_dir, train=False, download=True, transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bsz, shuffle=True, generator=g, num_workers=1)
    return testloader


def food101_train(log_dir, world_size, trainer_rank, train_bsz, seed, num_workers=1):
    helper.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                 std=[0.229, 0.224, 0.225])])
    trainset = torchvision.datasets.Food101(root=log_dir, split='train', transform=transform, download=True)
    partition = DataPartioner(trainset, world_size)
    partition = partition.use(trainer_rank)
    trainloader = torch.utils.data.DataLoader(partition, batch_size=train_bsz, shuffle=True,
                                              worker_init_fn=helper.set_seed(seed), generator=g,
                                              num_workers=num_workers)
    return trainloader, len(trainset)


def food101_test(log_dir, test_bsz, seed):
    helper.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    transform = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testset = torchvision.datasets.Food101(root=log_dir, split='test', transform=transform, download=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=test_bsz, shuffle=True, generator=g, num_workers=1)
    return testloader


# dir: directory where caltech dataset is downloaded
def caltech_classes(dir='/101_ObjectCategories', dataset_name='CalTech101'):
    if dataset_name == 'CalTech101':
        datadir = os.path.join(dir, '101_ObjectCategories')
    elif dataset_name == 'CalTech256':
        datadir = os.path.join(dir, '256_ObjectCategories')

    classes = []
    dirclasses = os.listdir(datadir)
    for d in dirclasses:
        classes.append(d)

    print(f'total classes in caltech256 {len(classes)}')
    return classes


def list_files(path):
    files = os.listdir(path)
    return np.asarray(files)


def split_files(datadir, out_dir, classes):
    for name in classes:
        full_dir = os.path.join(datadir, name)
        files = list_files(full_dir)
        total_file = np.size(files,0)

        train_size = math.ceil(total_file * 0.8)
        test_size = math.ceil(total_file * 0.2)

        train = files[0:train_size]
        test = files[train_size:]

        move_files(train, full_dir, out_dir, f"train/{name}")
        move_files(test, full_dir, out_dir, f"test/{name}")


def move_files(files, old_dir, out_dir, new_dir):
    new_dir = os.path.join(out_dir, new_dir)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    for file in np.nditer(files):
        old_file_path = os.path.join(old_dir, f'{file}')
        new_file_path = os.path.join(new_dir, f'{file}')

        shutil.copy(old_file_path, new_file_path)


def caltech_train(train_dir, bsz, world_size, rank, seed):
    helper.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    size = (224, 256)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(size=size[1]), transforms.CenterCrop(size=size[0]),
                                    transforms.ToTensor(), normalize])

    dataset = datasets.ImageFolder(train_dir, transform=transform)

    partition = DataPartioner(dataset, world_size)
    del dataset
    partition = partition.use(rank)

    trainloader = torch.utils.data.DataLoader(partition, batch_size=bsz, shuffle=True,
                                              worker_init_fn=helper.set_seed(seed), generator=g, num_workers=1)
    return trainloader, len(trainloader) * bsz * world_size


def caltech_test(testdir, test_bsz, seed):
    helper.set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    size = (224, 256)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize(size=size[1]), transforms.CenterCrop(size=size[0]),
                                    transforms.ToTensor(), normalize])

    dataset = datasets.ImageFolder(testdir, transform=transform)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=test_bsz, shuffle=False, generator=g, num_workers=1)

    return testloader


if __name__ == '__main__':
    # (CalTech101: https://data.caltech.edu/records/mzrjq-6wc02)
    # (CalTech256: https://data.caltech.edu/records/nyy15-4j048)
    # change logdir and dataset-name to where Caltech 101/256 data is downloaded
    logdir, dataset_name = '/101_ObjectCategories', 'CalTech101'
    classes = caltech_classes(dir=logdir, dataset_name=dataset_name)
    split_files(datadir=logdir, out_dir='/caltech101', classes=classes)

    # logdir, dataset_name = '/256_ObjectCategories', 'CalTech256'
    # classes = caltech_classes(dir=logdir, dataset_name=dataset_name)
    # split_files(datadir=logdir, out_dir='/caltech256', classes=classes)