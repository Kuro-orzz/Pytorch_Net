# Paper: https://arxiv.org/pdf/1911.11929

# How to split into 2 parts
# https://github.com/WongKinYiu/CrossStagePartialNetworks/issues/18

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
    
class CSPNet(nn.Module):
    def __init__(
            self, block_type: type[nn.Module]=Bottleneck,
            nblocks: list[int] = [6, 12, 24, 16],
            growth_rate: int=32, reduction: float=0.5, num_classes: int=10,
            split_ratio: float=0.5,
    ) -> None:
        super(CSPNet, self).__init__()
        self.growth_rate = growth_rate
        self.split_ratio = split_ratio

        num_planes = 2 * growth_rate
        # 3 is number of channel of img
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=7, stride=2, padding=1, bias=False)

        part1 = int(num_planes * self.split_ratio)
        part2 = num_planes - part1
        self.denseblock1 = self._make_dense_layers(block_type, part2, nblocks[0])
        num_planes = part2 + nblocks[0] * growth_rate + part1 
        out_planes = int(math.floor(num_planes * reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        part1 = int(num_planes * self.split_ratio)
        part2 = num_planes - part1
        self.denseblock2 = self._make_dense_layers(block_type, part2, nblocks[1])
        num_planes = part2 + nblocks[1] * growth_rate + part1
        out_planes = int(math.floor(num_planes * reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        part1 = int(num_planes * self.split_ratio)
        part2 = num_planes - part1
        self.denseblock3 = self._make_dense_layers(block_type, part2, nblocks[2])
        num_planes = part2 + nblocks[2] * growth_rate + part1
        out_planes = int(math.floor(num_planes * reduction))
        self.trans3 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.denseblock4 = self._make_dense_layers(block_type, num_planes, nblocks[3])
        num_planes += nblocks[3] * growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.fc = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_channels: int, nblock: int):
        layers = []
        for _ in range(nblock):
            layers.append(block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)
    
    def split_feature(self, X):
        channel = X.size(1)
        mid = int(channel * self.split_ratio)      
        return X[:, :mid, :, :], X[:, mid:, :, :]

    def forward(self, x):
        out = self.conv1(x)
        x1, out = self.split_feature(out)
        out = self.trans1(torch.cat([self.denseblock1(out), x1], dim=1))
        x1, out = self.split_feature(out)
        out = self.trans2(torch.cat([self.denseblock2(out), x1], dim=1))
        x1, out = self.split_feature(out)
        out = self.trans3(torch.cat([self.denseblock3(out), x1], dim=1))
        out = self.denseblock4(out)
        # kernel size must smaller than img size, so min img must be 128x128
        out = F.avg_pool2d(F.relu(self.bn(out)), 7)
        out = out.view(out.size(dim=0), -1)
        out = F.softmax(out, dim=1)
        return out


if __name__ == '__main__':
    net = CSPNet()
    x = torch.randn(1, 3, 128, 128)
    y = net(x)
    print(y)