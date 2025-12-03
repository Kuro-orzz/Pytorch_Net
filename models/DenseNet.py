import math, torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate) -> None:
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out, x], 1)
        return out
    
class Transition(nn.Module):
    def __init__(self, in_channels, output_channels) -> None:
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, output_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out
    
class DenseNet(nn.Module):
    def __init__(
            self, block_type: type[nn.Module]=Bottleneck,
            nblocks: list[int] = [6, 12, 24, 16],
            growth_rate: int=32, reduction: float=0.5, num_classes: int=10,
    ) -> None:
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate

        num_planes = 2 * growth_rate
        # 3 is number of channel of img
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=7, stride=2, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block_type, num_planes, nblocks[0])
        num_planes += nblocks[0] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block_type, num_planes, nblocks[1])
        num_planes += nblocks[1] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block_type, num_planes, nblocks[2])
        num_planes += nblocks[2] * growth_rate
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(block_type, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_channels: int, nblock: int):
        layers = []
        for _ in range(nblock):
            layers.append(block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # if x.shape == (128, 128, 3)
        # 128 -> 64
        out = self.conv1(x)
        # 64 -> 32
        out = self.trans1(self.dense1(out))
        # 32 -> 16
        out = self.trans2(self.dense2(out))
        # 16 -> 8
        out = self.trans3(self.dense3(out))
        out = self.dense4(out)
        # kernel size must smaller than img size, so min img must be 128x128
        out = F.avg_pool2d(F.relu(self.bn(out)), 7)
        out = out.view(out.size(dim=0), -1)
        out = self.linear(out)
        return out
    
def DenseNet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32)

def DenseNet264():
    return DenseNet(Bottleneck, [6, 12, 64, 48], growth_rate=32)

def DenseNet_CIFAR():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)


if __name__ == '__main__':
    net = DenseNet121()
    x = torch.randn(1, 3, 128, 128)
    y = net(x)
    print(y)