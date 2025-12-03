import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSoftAttention(nn.Module):
    def __init__(self) -> None:
        super(ChannelSoftAttention, self).__init__()

    def forward(self, m):
        B, C, H, W = m.shape
        # Step 1: z = avp(m) -> reshape to [B, C]
        z = F.adaptive_avg_pool2d(m, 1).view(B, C)
        # Step 2: softmax(z)
        alpha = F.softmax(z, dim=1)
        # Step 3: reshape [B, C, 1, 1] để broadcast
        alpha = alpha.view(B, C, 1, 1)
        # Step 4: weighted feature = alpha * m
        weighted = m * alpha
        # Step 5: sum all channel -> [B, 1, H, W]
        att = weighted.sum(dim=1, keepdim=True)
        return att

class BasicConv(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 kernel_size=3, stride=1, padding=1
    ) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        return F.relu(out)

class CNNet(nn.Module):
    def __init__(self, num_channels, output_channels) -> None:
        super(CNNet, self).__init__()
        
        self.blk1 = nn.Sequential(
            BasicConv(num_channels, 16),
            nn.MaxPool2d(2),
        )
        self.blk2 = nn.Sequential(
            BasicConv(16, 32),
            nn.MaxPool2d(2),
        )
        self.blk3 = nn.Sequential(
            BasicConv(32, 64),
            nn.MaxPool2d(2),
        )
        self.blk4 = nn.Sequential(
            BasicConv(64, 128),
            nn.MaxPool2d(2),
        )
        self.att = ChannelSoftAttention()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, output_channels)

    def forward(self, x):
        x = self.blk4(self.blk3(self.blk2(self.blk1(x))))
        x = self.att(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        out = torch.sigmoid(self.fc2(x))
        return out
        
if __name__ == '__main__':
    
    net = CNNet(3, 8)
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y)