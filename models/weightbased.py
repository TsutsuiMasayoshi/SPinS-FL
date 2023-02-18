import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math
import numpy as np
import random

__all__ = ['Conv6']

class KaimingConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        nn.Parameter(self.weight)

    def forward(self, x):
        x = F.conv2d(
            x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class KaimingLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")
        nn.Parameter(self.weight)
        nn.Parameter(self.bias)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class Conv6(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = KaimingConv(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.conv2 = KaimingConv(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = KaimingConv(64, 128, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.conv4 = KaimingConv(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128, affine=False)
        self.conv5 = KaimingConv(128, 256, 3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256, affine=False)
        self.conv6 = KaimingConv(256, 256, 3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256, affine=False)
        self.fc1 = KaimingLinear(4*4*256, 256, bias=False)
        self.bn7 = nn.BatchNorm1d(num_features=256, affine=False)
        self.fc2 = KaimingLinear(256, 256, bias=False)
        self.bn8 = nn.BatchNorm1d(num_features=256, affine=False)
        self.fc3 = KaimingLinear(256, 10, bias=False)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn7(self.fc1(x)))
        x = F.relu(self.bn8(self.fc2(x)))
        output = self.fc3(x)
        return output

if __name__ == '__main__':
    model = Conv6()