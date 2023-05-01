import torch
import torch.nn as nn
from torch.nn import Parameter
from kornia.enhance.zca import ZCAWhitening


class ZCA:
    def __init__(self):
        self.zca = ZCAWhitening(detach_transforms=False)

    def __call__(self, x):
        return self.zca(x, include_fit=True)


class DBN(nn.Module):
    """
    Realization is taken from:
    https://github.com/huangleiBuaa/IterNorm-pytorch/blob/master/extension/normailzation/dbn.py
    which in turn is based on the original paper implementation:
    https://github.com/princeton-vl/DecorrelatedBN/blob/master/module/DecorelateBN_NoAlign.lua

    Implementation is adapted for 2D tensor with dim=1 being number of features and dim=0 being number of samples.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()

        # Store values
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # Register params of additional linear layer
        if self.affine:
            self.weight = Parameter(torch.Tensor(1, num_features))
            self.bias = Parameter(torch.Tensor(1, num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_features, 1))
        self.register_buffer('running_projection', torch.eye(num_features))
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            nn.init.uniform_(self.weight)
            nn.init.zeros_(self.bias)

    def __call__(self, input_x):
        x = input_x.transpose(0, 1)

        if self.training:
            # Calculate mean of batch, update running_mean and subtract itt
            mean = x.mean(dim=1, keepdim=True)
            self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
            x_mean = x - mean

            # Calculate covariance matrix and
            sigma = x_mean.matmul(x_mean.t()) / x.size(1) + self.eps * torch.eye(self.num_features,
                                                                                 device=input_x.device)

            # SVD decomposition
            u, eig, _ = sigma.svd()

            # Apply 1/sqrt(x) for each element of tensor
            scale = eig.rsqrt()

            # Calculate and update running_projection matrix
            wm = u.matmul(scale.diag()).matmul(u.t())
            self.running_projection = (1. - self.momentum) * self.running_projection + self.momentum * wm

            # Apply running_projection matrix
            y = wm.matmul(x_mean)

        else:
            # Apply found values for running_projection and running_mean during inference
            x_mean = x - self.running_mean
            y = self.running_projection.matmul(x_mean)

        output = y.transpose(0, 1)

        # Additional affine transform
        if self.affine:
            output = output * self.weight + self.bias

        return output


class NoWhitening:
    def __init__(self):
        pass

    def __call__(self, x):
        return x
