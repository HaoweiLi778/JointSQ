import torch, math

""" DenseNet-40 """

norm_mean, norm_var = 0.0, 1.0

cov_cfg=[(3*i+1) for i in range(12*3+2+1)]


class DenseBasicBlock(torch.nn.Module):
    def __init__(self, inplanes, outplanes, dropRate=0):
        super(DenseBasicBlock, self).__init__()

        self.bn1 = torch.nn.BatchNorm2d(inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv1 = torch.nn.Conv2d(inplanes, outplanes, kernel_size=3,
                               padding=1, bias=False)

        self.dropRate = dropRate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.dropRate > 0:
            out = torch.nn.functional.dropout(out, p=self.dropRate, training=self.training)

        out = torch.cat((x, out), 1)

        return out


class Transition(torch.nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Transition, self).__init__()
        self.bn1 = torch.nn.BatchNorm2d(inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv1 = torch.nn.Conv2d(inplanes, outplanes, kernel_size=1,
                               bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = torch.nn.functional.avg_pool2d(out, 2)
        return out


class CifarDenseNet(torch.nn.Module):

    def __init__(self, depth=40, block=DenseBasicBlock, dropRate=0, num_classes=10, growthRate=12, compressionRate=1):
        super(CifarDenseNet, self).__init__()

        assert (depth - 4) % 3 == 0, 'depth should be 3n+4'
        n = (depth - 4) // 3 if 'DenseBasicBlock' in str(block) else (depth - 4) // 6

        transition = Transition

        self.covcfg=cov_cfg

        self.growthRate = growthRate
        self.dropRate = dropRate

        self.inplanes = growthRate * 2
        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_denseblock(block, n)
        self.trans1 = self._make_transition(transition, compressionRate)
        self.dense2 = self._make_denseblock(block, n)
        self.trans2 = self._make_transition(transition, compressionRate)
        self.dense3 = self._make_denseblock(block, n)
        self.bn = torch.nn.BatchNorm2d(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.avgpool = torch.nn.AvgPool2d(8)

        self.fc = torch.nn.Linear(self.inplanes, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_denseblock(self, block, num_block):
        layers = []
        for i in range(num_block):
            layers.append(block(self.inplanes, outplanes=self.growthRate, dropRate=self.dropRate))
            self.inplanes += self.growthRate

        return torch.nn.Sequential(*layers)

    def _make_transition(self, transition, compressionRate):
        inplanes = self.inplanes
        outplanes = int(math.floor(self.inplanes // compressionRate))
        self.inplanes = outplanes
        return transition(inplanes, outplanes)

    def forward(self, x):
        x = self.conv1(x)

        x = self.dense1(x)
        x = self.trans1(x)
        x = self.dense2(x)
        x = self.trans2(x)
        x = self.dense3(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def densenet40():
    return CifarDenseNet(depth=40, block=DenseBasicBlock)


# -----------------------------------
# FOR TEST
# -----------------------------------
if __name__ == "__main__":
    from thop import profile

    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_image = torch.randn(1, 3, 32, 32).to(device)

    model = densenet40().to(device)
    print(model)

    flops, params = profile(model, inputs=(input_image,))

    print('Params: %.2f'%(params / 10**6))
    print('Flops: %.2f'%(flops / 10**6))

    ret = model(input_image)
    
    print(ret.shape)

    print(".......")

