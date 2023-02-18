import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math
import numpy as np
import random

__all__ = ['SupermaskConv6']

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get supermask by sorting the scores and using the top k% of them
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # generating supermask from sorted scores
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


class SupermaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        self.sparsity = kwargs.pop('sparsity')
        super().__init__(*args, **kwargs)

        # score initialization
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        n = args[0] * args[2] * args[2] #input-channel * filtersize

        kaimingStdev = np.sqrt(2. / n)
        # NOTE: weight values takes either of {-sigma, sigma}, simga is stdev of kaiming distribution.
        # Because it performed well on previous research
        weightVals = np.array([kaimingStdev * random.choice([-1, 1]) for _ in range(self.weight.numel())])
        self.weight = nn.Parameter(torch.tensor(weightVals, dtype=torch.float32).reshape(self.weight.size()))

        # NOTE: Fix the weights at their initialization
        self.weight.requires_grad = False

    def forward(self, x):
        # NOTE: subnet is dicided based on the 'abs' of scores according to the Edge-Popup Algorithm
        subnet = GetSubnet.apply(self.scores.abs(), self.sparsity)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        self.sparsity = kwargs.pop('sparsity')
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        n = args[0] # number of input features
        stdev = np.sqrt(2. / n)
        stdevs = np.array([stdev * random.choice([-1, 1]) for _ in range(self.weight.numel())])
        self.weight = nn.Parameter(torch.tensor(stdevs, dtype=torch.float32).reshape(self.weight.size()))

        # NOTE: Fix the weights at their initialization
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores.abs(), self.sparsity)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)


class SupermaskConv6(nn.Module):
    def __init__(self, sparsity: float):
        super().__init__()
        self.conv1 = SupermaskConv(3, 64, 3, stride=1, padding=1, bias=False, sparsity=sparsity)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.conv2 = SupermaskConv(64, 64, 3, stride=1, padding=1, bias=False, sparsity=sparsity)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = SupermaskConv(64, 128, 3, stride=1, padding=1, bias=False, sparsity=sparsity)
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.conv4 = SupermaskConv(128, 128, 3, stride=1, padding=1, bias=False, sparsity=sparsity)
        self.bn4 = nn.BatchNorm2d(128, affine=False)
        self.conv5 = SupermaskConv(128, 256, 3, stride=1, padding=1, bias=False, sparsity=sparsity)
        self.bn5 = nn.BatchNorm2d(256, affine=False)
        self.conv6 = SupermaskConv(256, 256, 3, stride=1, padding=1, bias=False, sparsity=sparsity)
        self.bn6 = nn.BatchNorm2d(256, affine=False)
        self.fc1 = SupermaskLinear(4*4*256, 256, bias=False, sparsity=sparsity)
        self.bn7 = nn.BatchNorm1d(num_features=256, affine=False)
        self.fc2 = SupermaskLinear(256, 256, bias=False, sparsity=sparsity)
        self.bn8 = nn.BatchNorm1d(num_features=256, affine=False)
        self.fc3 = SupermaskLinear(256, 10, bias=False, sparsity=sparsity)

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
    model = SupermaskConv6(0.5)