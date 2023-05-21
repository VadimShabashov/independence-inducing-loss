import torch
import torch.nn as nn
from torch.nn import Parameter


class ZCA1(nn.Module):
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

    def __init__(self, device, num_features, groups=1, eps=1e-4, momentum=0.1, affine=True):
        super().__init__()

        self.device = device
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.groups = groups
        self.weight = Parameter(torch.Tensor(num_features, 1)).to(device=self.device)
        self.bias = Parameter(torch.Tensor(num_features, 1)).to(device=self.device)
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

            if self.affine:
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

            if self.affine:
                x = x * self.weight + self.bias

            x = x.view(C, N, H, W).transpose(0, 1)

            # Get 2D from 4D
            x = x[:, :, 0, 0]

            return x

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def _load_from_state_dict(self, state_dict, prefix, metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super(ZCA1, self)._load_from_state_dict(
            state_dict, prefix, metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )


class ZCA2(torch.nn.Module):
    """
    Implementation of IterNorm. It's taken from:
    https://github.com/huangleiBuaa/IterNorm-pytorch
    """

    class IterativeNormalization(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args, **kwargs):
            X, running_mean, running_wmat, nc, ctx.T, eps, momentum, training = args
            # change NxCxHxW to (G x D) x(NxHxW), i.e., g*d*m
            ctx.g = X.size(1) // nc
            x = X.transpose(0, 1).contiguous().view(ctx.g, nc, -1)
            _, d, m = x.size()
            saved = []

            if training:
                # calculate centered activation by subtracted mini-batch mean
                mean = x.mean(-1, keepdim=True)
                xc = x - mean
                saved.append(xc)
                # calculate covariance matrix
                P = [None] * (ctx.T + 1)
                P[0] = torch.eye(d).to(X).expand(ctx.g, d, d)
                Sigma = torch.baddbmm(eps, P[0], 1. / m, xc, xc.transpose(1, 2))
                # reciprocal of trace of Sigma: shape [g, 1, 1]
                rTr = (Sigma * P[0]).sum((1, 2), keepdim=True).reciprocal_()
                saved.append(rTr)
                Sigma_N = Sigma * rTr
                saved.append(Sigma_N)
                for k in range(ctx.T):
                    P[k + 1] = torch.baddbmm(1.5, P[k], -0.5, torch.matrix_power(P[k], 3), Sigma_N)
                saved.extend(P)
                wm = P[ctx.T].mul_(rTr.sqrt())  # whiten matrix: the matrix inverse of Sigma, i.e., Sigma^{-1/2}
                running_mean.copy_(momentum * mean + (1. - momentum) * running_mean)
                running_wmat.copy_(momentum * wm + (1. - momentum) * running_wmat)
            else:
                xc = x - running_mean
                wm = running_wmat
            xn = wm.matmul(xc)
            Xn = xn.view(X.size(1), X.size(0), *X.size()[2:]).transpose(0, 1).contiguous()
            ctx.save_for_backward(*saved)
            return Xn

        @staticmethod
        def backward(ctx, *grad_outputs):
            grad, = grad_outputs
            saved = ctx.saved_variables
            xc = saved[0]  # centered input
            rTr = saved[1]  # trace of Sigma
            sn = saved[2].transpose(-2, -1)  # normalized Sigma
            P = saved[3:]  # middle result matrix,
            g, d, m = xc.size()

            g_ = grad.transpose(0, 1).contiguous().view_as(xc)
            g_wm = g_.matmul(xc.transpose(-2, -1))
            g_P = g_wm * rTr.sqrt()
            wm = P[ctx.T]
            g_sn = 0
            for k in range(ctx.T, 1, -1):
                P[k - 1].transpose_(-2, -1)
                P2 = P[k - 1].matmul(P[k - 1])
                g_sn += P2.matmul(P[k - 1]).matmul(g_P)
                g_tmp = g_P.matmul(sn)
                g_P.baddbmm_(1.5, -0.5, g_tmp, P2)
                g_P.baddbmm_(1, -0.5, P2, g_tmp)
                g_P.baddbmm_(1, -0.5, P[k - 1].matmul(g_tmp), P[k - 1])
            g_sn += g_P
            # g_sn = g_sn * rTr.sqrt()
            g_tr = ((-sn.matmul(g_sn) + g_wm.transpose(-2, -1).matmul(wm)) * P[0]).sum((1, 2), keepdim=True) * P[0]
            g_sigma = (g_sn + g_sn.transpose(-2, -1) + 2. * g_tr) * (-0.5 / m * rTr)
            # g_sigma = g_sigma + g_sigma.transpose(-2, -1)
            g_x = torch.baddbmm(wm.matmul(g_ - g_.mean(-1, keepdim=True)), g_sigma, xc)
            grad_input = g_x.view(grad.size(1), grad.size(0), *grad.size()[2:]).transpose(0, 1).contiguous()
            return grad_input, None, None, None, None, None, None, None

    def __init__(self, device, num_features, num_groups=1, num_channels=None,
                 T=5, dim=4, eps=1e-5, momentum=0.1, affine=True):
        super(ZCA2, self).__init__()

        # assert dim == 4, 'IterNorm is not support 2D'
        self.T = T
        self.eps = eps
        self.momentum = momentum
        self.num_features = num_features
        self.affine = affine
        self.dim = dim
        self.device = device

        if num_channels is None:
            num_channels = (num_features - 1) // num_groups + 1

        num_groups = num_features // num_channels

        while num_features % num_channels != 0:
            num_channels //= 2
            num_groups = num_features // num_channels

        assert num_groups > 0 and num_features % num_groups == 0, "num features={}, num groups={}".format(num_features,
            num_groups)

        self.num_groups = num_groups
        self.num_channels = num_channels
        shape = [1] * dim
        shape[1] = self.num_features

        if self.affine:
            self.weight = Parameter(torch.Tensor(*shape)).to(device=self.device)
            self.bias = Parameter(torch.Tensor(*shape)).to(device=self.device)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.register_buffer('running_mean', torch.zeros(num_groups, num_channels, 1, device=self.device))
        # running whiten matrix
        self.register_buffer(
            'running_wm',
            torch.eye(num_channels, device=self.device).expand(num_groups, num_channels, num_channels).clone()
        )
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor):
        X_hat = self.IterativeNormalization.apply(
            X, self.running_mean, self.running_wm, self.num_channels, self.T,
            self.eps, self.momentum, self.training
        )

        # affine
        if self.affine:
            return X_hat * self.weight + self.bias
        else:
            return X_hat

    def extra_repr(self):
        return '{num_features}, num_channels={num_channels}, T={T}, eps={eps}, ' \
               'momentum={momentum}, affine={affine}'.format(**self.__dict__)


class NoWhitening:
    def __init__(self):
        pass

    def __call__(self, x):
        return x
