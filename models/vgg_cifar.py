import torch
import math

DEFAULT_CFG = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512],
}


class VGG(torch.nn.Module):

    # def __init__(self, num_cls=10, dataset='CIFAR10', depth=19, init_weights=True, batch_norm=True, cfg=None):
    def __init__(self, num_cls=21, dataset='CIFAR10',depth=19, init_weights=True, batch_norm=True, cfg=None):
        super(VGG, self).__init__()

        if dataset == 'CIFAR10':
            self.num_classes = 10
        # if dataset == 'CIFAR100':
        #    self.num_classes = 100
        self.num_classes = num_cls
        self.cfg = cfg if cfg else cfg[depth]
        self.features = self.__make_layers(self.cfg, batch_norm=batch_norm)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(self.cfg[-1], 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, self.num_classes),
        )
        if init_weights:
            self.__initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def __make_layers(self, cfg, batch_norm=False) -> torch.nn.Sequential:
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, torch.nn.BatchNorm2d(v), torch.nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, torch.nn.ReLU(inplace=True)]
                in_channels = v
        return torch.nn.Sequential(*layers)

    def __initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, torch.nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def vgg11(dataset='CIFAR10', **kwargs) -> VGG:
    """VGG 11-layer model

    Args:
        dataset: CIFAR10/CIFAR100
    """
    return VGG(dataset=dataset,
               depth=11,
               batch_norm=False,
               init_weights=True,
               cfg=DEFAULT_CFG[11],
               **kwargs)


def vgg11_bn(dataset='CIFAR10', **kwargs) -> VGG:
    """VGG 11-layer model with batch normalization

    Args:
        dataset: CIFAR10/CIFAR100
    """
    return VGG(dataset=dataset,
               depth=11,
               batch_norm=True,
               init_weights=True,
               cfg=DEFAULT_CFG[11],
               **kwargs)


def vgg13(dataset='CIFAR10', **kwargs) -> VGG:
    """VGG 13-layer model 

    Args:
        dataset: CIFAR10/CIFAR100
    """
    return VGG(dataset=dataset,
               depth=13,
               batch_norm=False,
               init_weights=True,
               cfg=DEFAULT_CFG[13],
               **kwargs)


def vgg13_bn(dataset='CIFAR10', **kwargs) -> VGG:
    """VGG 13-layer model with batch normalization

    Args:
        dataset: CIFAR10/CIFAR100
    """
    return VGG(dataset=dataset,
               depth=13,
               batch_norm=True,
               init_weights=True,
               cfg=DEFAULT_CFG[13],
               **kwargs)


def vgg16(dataset='CIFAR10', **kwargs) -> VGG:
    """VGG 16-layer model

    Args:
        dataset: CIFAR10/CIFAR100
    """
    return VGG(dataset=dataset,
               depth=16,
               batch_norm=False,
               init_weights=True,
               cfg=DEFAULT_CFG[16],
               **kwargs)


# def vgg16_bn(num_classes=21, dataset='CIFAR10', **kwargs) -> VGG:
def vgg16_bn(num_classes=21, **kwargs) -> VGG:
    """VGG 16-layer model with batch normalization

    Args:
        dataset: CIFAR10/CIFAR100
    """

    model = VGG(num_classes,
                # dataset=dataset,
                depth=16,
                batch_norm=True,
                init_weights=True,
                cfg=DEFAULT_CFG[16],
                **kwargs)
    return model


def vgg19(dataset='CIFAR10', **kwargs) -> VGG:
    """VGG 19-layer model 

    Args:
        dataset: CIFAR10/CIFAR100
    """
    return VGG(dataset=dataset,
               depth=19,
               batch_norm=False,
               init_weights=True,
               cfg=DEFAULT_CFG[19],
               **kwargs)


def vgg19_bn(num_classes=10, dataset='CIFAR10', **kwargs) -> VGG:
    """VGG 19-layer model with batch normalization

    Args:
        dataset: CIFAR10/CIFAR100
    """
    return VGG(num_classes,
               dataset=dataset,
               depth=19,
               batch_norm=True,
               init_weights=True,
               cfg=DEFAULT_CFG[19],
               **kwargs)


if __name__ == '__main__':
    model = vgg16_bn(dataset="CIFAR10")
    print(model)
    for _, layer in enumerate(model.parameters()):
        print(layer)
    x = torch.autograd.Variable(torch.FloatTensor(16, 3, 40, 40))
    y = model(x)
    print(y.data.shape)
