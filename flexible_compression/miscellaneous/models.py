from torch import nn, optim
import torchvision.models as models

import flexible_compression.miscellaneous.helper as helper


def get_model(model_name, determinism, args):
    if model_name == 'resnet50':
        model_obj = ResNet50Object(lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, seed=args.seed,
                                   gamma=args.gamma, determinism=determinism)

    elif model_name == 'resnet18':
        model_obj = ResNet18Object(lr=args.lr, momentum=args.momentum, seed=args.seed, weight_decay=args.weight_decay,
                                   gamma=args.gamma, determinism=determinism)

    elif model_name == 'alexnet':
        model_obj = AlexNetObject(lr=args.lr, seed=args.seed, gamma=args.gamma, momentum=args.momentum,
                                  weightdecay=args.weight_decay, determinism=determinism)

    elif model_name == 'vision_transformer':
        model_obj = VisionTransformer(lr=args.lr, momentum=args.momentum, seed=args.seed, weightdecay=args.weight_decay,
                                      gamma=args.gamma, determinism=determinism)

    return model_obj


class ResNet18Object(object):
    def __init__(self, lr, momentum, weight_decay, seed, gamma, determinism):
        helper.set_seed(seed, determinism)
        self.lr = lr
        self.momentum = momentum
        self.weightdecay = weight_decay
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()
        self.model = models.resnet18(progress=True, pretrained=False)
        self.optim = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weightdecay)
        milestones = [15, 30, 45]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optim, milestones=milestones, gamma=self.gamma,
                                                           last_epoch=-1)

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.optim

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        return self.lr_scheduler


class ResNet50Object(object):
    def __init__(self, lr, momentum, weight_decay, seed, gamma, determinism):
        helper.set_seed(seed, determinism)
        self.lr = lr
        self.momentum = momentum
        self.weightdecay = weight_decay
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()
        self.model = models.resnet50(progress=True, pretrained=False)
        self.optim = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weightdecay)
        milestones = [100, 150, 200]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optim, milestones=milestones, gamma=self.gamma,
                                                           last_epoch=-1)

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.optim

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        return self.lr_scheduler


class AlexNetObject(object):
    def __init__(self, lr, gamma, seed, momentum, weightdecay, determinism):
        helper.set_seed(seed, determinism)
        self.lr = lr
        self.gamma = gamma
        self.momentum = momentum
        self.weightdecay = weightdecay
        self.loss = nn.CrossEntropyLoss()
        self.model = models.alexnet(progress=True, pretrained=False)
        self.opt = optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weightdecay)
        milestones = [25, 50, 75]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.opt, milestones=milestones, gamma=self.gamma,
                                                           last_epoch=-1)

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.opt

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        return self.lr_scheduler


class VisionTransformer(object):
    def __init__(self, lr, gamma, momentum, weightdecay, seed, determinism):
        helper.set_seed(seed, determinism)
        self.lr = lr
        self.gamma = gamma
        self.momentum = momentum
        self.weightdecay = weightdecay
        self.loss = nn.CrossEntropyLoss()
        self.model = models.vit_l_16(progress=False, pretrained=False)
        self.opt = optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weightdecay, betas=(0.9,0.999))
        milestones = [50, 100, 150]
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.opt, milestones=milestones, gamma=self.gamma, last_epoch=-1)

    def get_model(self):
        return self.model

    def get_optim(self):
        return self.opt

    def get_loss(self):
        return self.loss

    def get_lrscheduler(self):
        return self.lr_scheduler