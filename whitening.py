import torch
import torch.nn as nn
from torch.nn import Parameter


class ZCA(nn.Module):
    """
    Implementation is taken from:
    https://github.com/cvlab-epfl/Power-Iteration-SVD
    """

    class PowerIterationOnce(torch.autograd.Function):
        @staticmethod
        def forward(ctx, M, v_k, num_iter=19):
            """
            :param ctx: used to save materials for backward.
            :param M: n by n matrix.
            :param v_k: initial guess of leading vector.
            :return: v_k1 leading vector.
            """
            ctx.num_iter = num_iter
            ctx.save_for_backward(M, v_k)
            return v_k

        @staticmethod
        def backward(ctx, grad_output):
            M, v_k = ctx.saved_tensors
            dL_dvk = grad_output
            I = torch.eye(M.shape[-1], out=torch.empty_like(M))
            numerator = I - v_k.mm(torch.t(v_k))
            denominator = torch.norm(M.mm(v_k)).clamp(min=1.e-5)
            ak = numerator / denominator
            term1 = ak
            q = M / denominator
            for i in range(1, ctx.num_iter + 1):
                ak = q.mm(ak)
                term1 += ak
            dL_dM = torch.mm(term1.mm(dL_dvk), v_k.t())
            return dL_dM, ak

    def __init__(self, num_features, device, groups=1, eps=1e-4, momentum=0.1, affine=True):
        super().__init__()

        self.device = device
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.zeros(num_features, 1, device=self.device))
        self.bias = Parameter(torch.zeros(num_features, 1, device=self.device))
        self.power_layer = self.PowerIterationOnce.apply
        self.register_buffer('running_mean', torch.zeros(num_features, 1, device=self.device))
        self.create_dictionary()
        self.reset_parameters()
        self.dict = self.state_dict()

    def create_dictionary(self):
        length = int(self.num_features / self.groups)
        for i in range(self.groups):
            self.register_buffer("running_subspace{}".format(i), torch.eye(length, length, device=self.device))
            for j in range(length):
                self.register_buffer('eigenvector{}-{}'.format(i, j), torch.ones(length, 1, device=self.device))

    def reset_running_stats(self):
        self.running_mean.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    @staticmethod
    def _check_input_dim(x):
        if x.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.dim()))

    def forward(self, x):
        # Transform 2D to 4D
        x = x[:, :, None, None]

        self._check_input_dim(x)
        if self.training:
            N, C, H, W = x.size()
            G = self.groups
            x = x.transpose(0, 1).contiguous().view(C, -1)
            mu = x.mean(1, keepdim=True)
            x = x - mu
            xxt = torch.mm(x, x.t()) / (N * H * W) + torch.eye(C, out=torch.empty_like(x), device=self.device) * self.eps

            assert C % G == 0
            length = int(C/G)
            xxti = torch.chunk(xxt, G, dim=0)
            xxtj = [torch.chunk(xxti[j], G, dim=1)[j] for j in range(G)]

            xg = list(torch.chunk(x, G, dim=0))

            xgr_list = []
            for i in range(G):
                counter_i = 0
                # compute eigenvectors of subgroups no grad
                with torch.no_grad():
                    u, e, v = torch.svd(xxtj[i])
                    ratio = torch.cumsum(e, 0)/e.sum()
                    for j in range(length):
                        if ratio[j] >= (1 - self.eps) or e[j] <= self.eps:
                            break
                        eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                        eigenvector_ij.data = v[:, j][..., None].data
                        counter_i = j + 1

                # feed eigenvectors to Power Iteration Layer with grad and compute whitened tensor
                subspace = torch.zeros_like(xxtj[i])
                for j in range(counter_i):
                    eigenvector_ij = self.__getattr__('eigenvector{}-{}'.format(i, j))
                    eigenvector_ij = self.power_layer(xxtj[i], eigenvector_ij)
                    lambda_ij = torch.mm(xxtj[i].mm(eigenvector_ij).t(), eigenvector_ij)/torch.mm(eigenvector_ij.t(), eigenvector_ij)
                    if lambda_ij < 0:
                        break
                    diff_ratio = (lambda_ij - e[j]).abs()/e[j]
                    if diff_ratio > 0.1:
                        break
                    subspace += torch.mm(eigenvector_ij, torch.rsqrt(lambda_ij).mm(eigenvector_ij.t()))
                    xxtj[i] = xxtj[i] - torch.mm(xxtj[i], eigenvector_ij.mm(eigenvector_ij.t()))
                xgr = torch.mm(subspace, xg[i])
                xgr_list.append(xgr)

                with torch.no_grad():
                    running_subspace = self.__getattr__('running_subspace' + str(i))
                    running_subspace.data = (1 - self.momentum) * running_subspace.data + self.momentum * subspace.data

            with torch.no_grad():
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mu

            xr = torch.cat(xgr_list, dim=0)
            xr = xr * self.weight + self.bias
            xr = xr.view(C, N, H, W).transpose(0, 1)

            # Get 2D from 4D
            xr = xr[:, :, 0, 0]

            return xr

        else:
            N, C, H, W = x.size()
            x = x.transpose(0, 1).contiguous().view(C, -1)
            x = (x - self.running_mean)
            G = self.groups
            xg = list(torch.chunk(x, G, dim=0))
            for i in range(G):
                subspace = self.__getattr__('running_subspace' + str(i))
                xg[i] = torch.mm(subspace, xg[i])
            x = torch.cat(xg, dim=0)
            x = x * self.weight + self.bias
            x = x.view(C, N, H, W).transpose(0, 1)

            # Get 2D from 4D
            x = x[:, :, 0, 0]

            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super(ZCA, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )


class DBN(nn.Module):
    """
    Realization is taken from:
    https://github.com/huangleiBuaa/IterNorm-pytorch/blob/master/extension/normailzation/dbn.py
    which in turn is based on the original paper implementation:
    https://github.com/princeton-vl/DecorrelatedBN/blob/master/module/DecorelateBN_NoAlign.lua

    Implementation is adapted for 2D tensor with dim=1 being number of features and dim=0 being number of samples.
    """

    def __init__(self, num_features, device, eps=1e-5, momentum=0.1, affine=False):
        super().__init__()

        # Store values
        self.device = device
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

        self.register_buffer('running_mean', torch.zeros(num_features, 1, device=self.device))
        self.register_buffer('running_projection', torch.eye(num_features, device=self.device))
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
                                                                                 device=self.device)

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
