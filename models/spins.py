import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import math
import numpy as np
import random

__all__ = ['SpinsConv6']

class GetSpinsSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, top_locked_indices, unlocked_indices, bottom_locked_indices):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        _, ind = flat_out[unlocked_indices].sort()
        ul_border = unlocked_indices.numel() // 2
        sink_indices = unlocked_indices[ind[:ul_border]]
        pop_indices = unlocked_indices[ind[ul_border:]]
        flat_out[top_locked_indices] = 1
        flat_out[pop_indices] = 1
        flat_out[sink_indices] = 0
        flat_out[bottom_locked_indices] = 0
        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None, None, None


class SpinsConvLayer(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        localPinRate = kwargs.pop('localPinRate')
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        half_locked_num = int((localPinRate / 2) * self.weight.numel())

        self.register_buffer('top_locked_indices', torch.zeros(half_locked_num, dtype=torch.long))
        self.register_buffer('unlocked_indices', torch.zeros(self.weight.numel() - 2 * half_locked_num, dtype=torch.long))
        self.register_buffer('bottom_locked_indices', torch.zeros(half_locked_num, dtype=torch.long))

        # NOTE: initialize the weights like this.
        n = args[0] * args[2] * args[2] #input-channel * filtersize
        stdev = np.sqrt(2. / n)
        stdevs = np.array([stdev * random.choice([-1, 1]) for _ in range(self.weight.numel())])
        self.weight = nn.Parameter(torch.tensor(stdevs, dtype=torch.float32).reshape(self.weight.size()))

        # NOTE: Fix the weights at their initialization
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSpinsSubnet.apply(self.scores.abs(), self.top_locked_indices, self.unlocked_indices, self.bottom_locked_indices)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x

class SpinsLinearLayer(nn.Linear):
    def __init__(self, *args, **kwargs):
        localPinRate = kwargs.pop('localPinRate')
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        half_locked_num = int((localPinRate / 2) * self.weight.numel())

        self.register_buffer('top_locked_indices', torch.zeros(half_locked_num, dtype=torch.long))
        self.register_buffer('unlocked_indices', torch.zeros(self.weight.numel() - 2 * half_locked_num, dtype=torch.long))
        self.register_buffer('bottom_locked_indices', torch.zeros(half_locked_num, dtype=torch.long))

        # NOTE: initialize the weights like this.
        n = args[0] # number of input features
        stdev = np.sqrt(2. / n)
        stdevs = np.array([stdev * random.choice([-1, 1]) for _ in range(self.weight.numel())])
        self.weight = nn.Parameter(torch.tensor(stdevs, dtype=torch.float32).reshape(self.weight.size()))

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False

    def forward(self, x):
        subnet = GetSpinsSubnet.apply(self.scores.abs(), self.top_locked_indices, self.unlocked_indices, self.bottom_locked_indices)
        w = self.weight * subnet
        return F.linear(x, w, self.bias)


class SpinsConv6(nn.Module):
    def __init__(self, localPinRate: float):
        super().__init__()
        self.conv1 = SpinsConvLayer(3, 64, 3, stride=1, padding=1, bias=False, localPinRate=localPinRate)
        self.bn1 = nn.BatchNorm2d(64, affine=False)
        self.conv2 = SpinsConvLayer(64, 64, 3, stride=1, padding=1, bias=False, localPinRate=localPinRate)
        self.bn2 = nn.BatchNorm2d(64, affine=False)
        self.conv3 = SpinsConvLayer(64, 128, 3, stride=1, padding=1, bias=False, localPinRate=localPinRate)
        self.bn3 = nn.BatchNorm2d(128, affine=False)
        self.conv4 = SpinsConvLayer(128, 128, 3, stride=1, padding=1, bias=False, localPinRate=localPinRate)
        self.bn4 = nn.BatchNorm2d(128, affine=False)
        self.conv5 = SpinsConvLayer(128, 256, 3, stride=1, padding=1, bias=False, localPinRate=localPinRate)
        self.bn5 = nn.BatchNorm2d(256, affine=False)
        self.conv6 = SpinsConvLayer(256, 256, 3, stride=1, padding=1, bias=False, localPinRate=localPinRate)
        self.bn6 = nn.BatchNorm2d(256, affine=False)
        self.fc1 = SpinsLinearLayer(4*4*256, 256, bias=False, localPinRate=localPinRate)
        self.bn7 = nn.BatchNorm1d(num_features=256, affine=False)
        self.fc2 = SpinsLinearLayer(256, 256, bias=False, localPinRate=localPinRate)
        self.bn8 = nn.BatchNorm1d(num_features=256, affine=False)
        self.fc3 = SpinsLinearLayer(256, 10, bias=False, localPinRate=localPinRate)

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
    model = SpinsConv6(0.75)