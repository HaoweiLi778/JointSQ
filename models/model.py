import math
import torch

""" LeNet """

class CifarLeNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(CifarLeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = torch.nn.Linear(16*5*5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def lenet():
    return CifarLeNet(num_classes=10)


""" AlexNet """

class CifarAlexNet(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(CifarAlexNet, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(64, 192, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(192, 384, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(384, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.Linear(256 * 2 * 2, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 2 * 2)
        x = self.classifier(x)
        return x

def alexnet():
    return CifarAlexNet(num_classes=10)


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




import torch


norm_mean, norm_var = 0.0, 1.0

cov_cfg=[(3*i+1) for i in range(12*3+2+1)]

""" GoogleNet """
# Set cp_rate to 1 if NOT compress.
class Inception(torch.nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes, tmp_name):
        super(Inception, self).__init__()
        self.tmp_name=tmp_name

        self.n1x1 = n1x1
        self.n3x3 = n3x3
        self.n5x5 = n5x5
        self.pool_planes = pool_planes

        # 1x1 conv branch
        if self.n1x1:
            conv1x1 = torch.nn.Conv2d(in_planes, n1x1, kernel_size=1)
            conv1x1.tmp_name = self.tmp_name

            self.branch1x1 = torch.nn.Sequential(
                conv1x1,
                torch.nn.BatchNorm2d(n1x1),
                torch.nn.ReLU(True),
            )

        # 1x1 conv -> 3x3 conv branch
        if self.n3x3:
            conv3x3_1=torch.nn.Conv2d(in_planes, n3x3red, kernel_size=1)
            conv3x3_2=torch.nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1)
            conv3x3_1.tmp_name = self.tmp_name
            conv3x3_2.tmp_name = self.tmp_name

            self.branch3x3 = torch.nn.Sequential(
                conv3x3_1,
                torch.nn.BatchNorm2d(n3x3red),
                torch.nn.ReLU(True),
                conv3x3_2,
                torch.nn.BatchNorm2d(n3x3),
                torch.nn.ReLU(True),
            )

        # 1x1 conv -> 5x5 conv branch
        if self.n5x5 > 0:
            conv5x5_1 = torch.nn.Conv2d(in_planes, n5x5red, kernel_size=1)
            conv5x5_2 = torch.nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1)
            conv5x5_3 = torch.nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1)
            conv5x5_1.tmp_name = self.tmp_name
            conv5x5_2.tmp_name = self.tmp_name
            conv5x5_3.tmp_name = self.tmp_name

            self.branch5x5 = torch.nn.Sequential(
                conv5x5_1,
                torch.nn.BatchNorm2d(n5x5red),
                torch.nn.ReLU(True),
                conv5x5_2,
                torch.nn.BatchNorm2d(n5x5),
                torch.nn.ReLU(True),
                conv5x5_3,
                torch.nn.BatchNorm2d(n5x5),
                torch.nn.ReLU(True),
            )

        # 3x3 pool -> 1x1 conv branch
        if self.pool_planes > 0:
            conv_pool = torch.nn.Conv2d(in_planes, pool_planes, kernel_size=1)
            conv_pool.tmp_name = self.tmp_name

            self.branch_pool = torch.nn.Sequential(
                torch.nn.MaxPool2d(3, stride=1, padding=1),
                conv_pool,
                torch.nn.BatchNorm2d(pool_planes),
                torch.nn.ReLU(True),
            )

    def forward(self, x):
        """
            branch1 = self.branch1(x)
            branch2 = self.branch2(x)
            branch3 = self.branch3(x)
            branch4 = self.branch4(x)
            outputs = [branch1, branch2, branch3, branch4]
            return outputs
        """
        out = []
        y1 = self.branch1x1(x)
        out.append(y1)

        y2 = self.branch3x3(x)
        out.append(y2)

        y3 = self.branch5x5(x)
        out.append(y3)

        y4 = self.branch_pool(x)
        out.append(y4)
        return torch.cat(out, 1)


class GoogLeNet(torch.nn.Module):
    def __init__(self, block=Inception, filters=None):
        super(GoogLeNet, self).__init__()

        first_outplanes=192
        conv_pre = torch.nn.Conv2d(3, first_outplanes, kernel_size=3, padding=1)
        conv_pre.tmp_name = 'pre_layer'
        self.pre_layers = torch.nn.Sequential(
            conv_pre,
            torch.nn.BatchNorm2d(first_outplanes),
            torch.nn.ReLU(True),
        )

        filters = [
            [64, 128, 32, 32],
            [128, 192, 96, 64],
            [192, 208, 48, 64],
            [160, 224, 64, 64],
            [128, 256, 64, 64],
            [112, 288, 64, 64],
            [256, 320, 128, 128],
            [256, 320, 128, 128],
            [384, 384, 128, 128]
        ]
        self.filters = filters

        mid_filters = [
            [96, 16],
            [128, 32],
            [96, 16],
            [112, 24],
            [128, 24],
            [144, 32],
            [160, 32],
            [160, 32],
            [192, 48]
        ]

        # cp_rate_list=[]
        # for cp_rate in compress_rate:
        #     cp_rate_list.append(1-cp_rate)

        in_plane_list=[]
        for i in range(8):
            in_plane_list.append(filters[i][0] +
                                    filters[i][1] + 
                                    filters[i][2] + 
                                    filters[i][3])


        self.inception_a3 = block(first_outplanes, 
                                    filters[0][0], mid_filters[0][0], filters[0][1], mid_filters[0][1], filters[0][2], filters[0][3], 
                                    'a3')
        self.inception_b3 = block(in_plane_list[0], 
                                    filters[1][0], mid_filters[1][0], filters[1][1], mid_filters[1][1], filters[1][2], filters[1][3], 
                                    'a4')
        self.maxpool1 = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.maxpool2 = torch.nn.MaxPool2d(3, stride=2, padding=1)
        self.inception_a4 = block(in_plane_list[1], 
                                    filters[2][0], mid_filters[2][0], filters[2][1], mid_filters[2][1], filters[2][2], filters[2][3], 
                                    'a4')
        self.inception_b4 = block(in_plane_list[2], 
                                    filters[3][0], mid_filters[3][0], filters[3][1], mid_filters[3][1], filters[3][2], filters[3][3], 
                                    'b4')
        self.inception_c4 = block(in_plane_list[3], 
                                    filters[4][0], mid_filters[4][0], filters[4][1], mid_filters[4][1], filters[4][2], filters[4][3], 
                                    'c4')
        self.inception_d4 = block(in_plane_list[4], 
                                    filters[5][0], mid_filters[5][0], filters[5][1], mid_filters[5][1], filters[5][2], filters[5][3], 
                                    'd4')
        self.inception_e4 = block(in_plane_list[5], 
                                    filters[6][0], mid_filters[6][0], filters[6][1], mid_filters[6][1], filters[6][2], filters[6][3], 
                                    'e4')
        self.inception_a5 = block(in_plane_list[6], 
                                    filters[7][0], mid_filters[7][0], filters[7][1], mid_filters[7][1], filters[7][2], filters[7][3], 
                                    'a5')
        self.inception_b5 = block(in_plane_list[7], 
                                    filters[8][0], mid_filters[8][0], filters[8][1], mid_filters[8][1], filters[8][2], filters[8][3], 
                                    'b5')
        self.avgpool = torch.nn.AvgPool2d(8, stride=1)
        self.linear = torch.nn.Linear(sum(filters[-1]), 10)

    def forward(self, x):

        out = self.pre_layers(x)

        # 192 x 32 x 32
        out = self.inception_a3(out)
        # 256 x 32 x 32
        out = self.inception_b3(out)
        # 480 x 32 x 32
        out = self.maxpool1(out)

        # 480 x 16 x 16
        out = self.inception_a4(out)
        # 512 x 16 x 16
        out = self.inception_b4(out)
        # 512 x 16 x 16
        out = self.inception_c4(out)
        # 512 x 16 x 16
        out = self.inception_d4(out)
        # 528 x 16 x 16
        out = self.inception_e4(out)
        # 823 x 16 x 16
        out = self.maxpool2(out)

        # 823 x 8 x 8
        out = self.inception_a5(out)
        # 823 x 8 x 8
        out = self.inception_b5(out)

        # 1024 x 8 x 8
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out

def googlenet():
    return GoogLeNet(block=Inception)


# -----------------------------------
# FOR TEST
# -----------------------------------
if __name__ == "__main__":
    from torchsummary import summary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input_image = torch.randn(1, 3, 32, 32).to(device)

    print("=================== lenet ========================\n"*3)
    model_lenet = lenet().to(device)
    summary(model_lenet, (3, 32, 32))
    print(model_lenet(input_image).shape)
    print("================== lenet =========================\n"*3)

    print("================= alexnet ==========================\n"*3)
    model_alexnet = alexnet().to(device)
    summary(model_alexnet, (3, 32, 32))
    print(model_alexnet(input_image).shape)
    print("=================== alexnet ========================\n"*3)


    print("=================== densenet40 ========================\n"*3)
    model_densenet = densenet40().to(device)
    summary(model_densenet, (3, 32, 32))
    print(model_densenet(input_image).shape)
    print("================= densenet40 ==========================\n"*3)


    print("================= googlenet ==========================\n"*3)
    model_googlenet = googlenet().to(device)
    summary(model_googlenet, (3, 32, 32))
    print(model_googlenet(input_image).shape)
    print("================ googlenet ===========================\n"*3)
    print(".......")


