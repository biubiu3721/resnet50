import torch.nn as nn
import torch


class Bottleneck(nn.Module):

    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=3, stride=1, bias=False, padding=1)
        self.bn1 = nn.BatchNorm2d(width)
        # resnet50 groups = 1
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 dropout_prob=0.7): # 添加drop out
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 16 # the more the better

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        last_out_channel = 128
        #  The numbers of filters are {16, 32, 64} respectively.
        # But Can't train a train error less than 15% with hyper params above. so I use , 128, 512
        self.layer1 = self._make_layer(block, 16, blocks_num[0], stride=2) # 64 is output channel, in channel = output channel * expansion in last blocks in each layer
        self.layer2 = self._make_layer(block, 32, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 64, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 128, blocks_num[2], stride=2)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.dropout = nn.Dropout(p=dropout_prob)             # # 添加drop out
            self.fc = nn.Linear(last_out_channel * block.expansion, num_classes)     #expansion = 4
        # init params
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion: # expansion = 4
            " The subsampling is performed by convolutions with a stride of 2."
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # max pool is not used in resnet on cifar-10
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)


        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.dropout(x)           # # 添加drop out
            x = self.fc(x)

        return x


# We conducted more studies on the CIFAR-10 dataset
# [20], which consists of 50k training images and 10k testing images in 10 classes.
# Then we use a stack of 6n layers with 3×3 convolutions on the feature maps of sizes {32, 16, 8} respectively,
# with 2n layers for each feature map size.
# We compare n = {3, 5, 7, 9}, leading to 20(conv + 6n + fc), 32, 44, and  56-layer networks. 
def resnet34(num_classes=10, include_top=True):
    return ResNet(Bottleneck, [3, 3, 3, 3], num_classes=num_classes, include_top=include_top)
