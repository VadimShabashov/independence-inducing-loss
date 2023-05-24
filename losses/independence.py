import torch

from correlation import non_diagonal_correlation


class BaseIndependenceLoss:
    """
    Base class for independence loss.
    """
    def __init__(self, device, alpha, eps):
        self.alpha = alpha
        self.eps = torch.tensor(eps, device=device)
        self.device = device


class NegApproxLoss1(BaseIndependenceLoss):
    """
    Minimization of mutual information using negentropy approximation from:
    https://www.cs.helsinki.fi/u/ahyvarin/papers/NIPS97.pdf

    Minimization of mutual information <=> maximization of negentropy <=> minimization of 1/negentropy
    """

    def __init__(self, device, alpha=10.0, eps=1e-8):
        super().__init__(device, alpha, eps)

    def __call__(self, x):
        # Find negentropy approximation for each feature
        k1 = 36 / (8 * torch.sqrt(torch.tensor(3)) - 9)
        k2a = 1 / (2 - 6 / torch.pi)

        d1 = x * torch.exp(- (x ** 2) / 2)
        d2 = torch.abs(x)

        neg_approx = k1 * d1.mean(axis=0) ** 2 + k2a * (d2.mean(axis=0) - torch.sqrt(torch.tensor(2 / torch.pi))) ** 2

        return self.alpha / max(neg_approx.mean(), self.eps)


class NegApproxLoss2(BaseIndependenceLoss):
    """
    Minimization of mutual information using negentropy approximation from:
    https://www.cs.helsinki.fi/u/ahyvarin/papers/NIPS97.pdf

    Minimization of mutual information <=> maximization of negentropy <=> minimization of 1/negentropy
    """

    def __init__(self, device, alpha=10.0, eps=1e-8):
        super().__init__(device, alpha, eps)

    def __call__(self, x):
        # Find negentropy approximation for each feature
        k1 = 36 / (8 * torch.sqrt(torch.tensor(3)) - 9)
        k2b = 24 / (16 * torch.sqrt(torch.tensor(3)) - 27)

        d1 = x * torch.exp(- (x ** 2) / 2)
        d2 = torch.exp(- (x ** 2) / 2)

        neg_approx = k1 * d1.mean(axis=0) ** 2 + k2b * (d2.mean(axis=0) - torch.sqrt(1 / torch.tensor(2))) ** 2

        return self.alpha / max(neg_approx.mean(), self.eps)


class KurtosisLoss(BaseIndependenceLoss):
    """
    Minimize non-gaussianity of output by maximizing kurtosis.
    Kurtosis maximization <=> minimization of 1/kurtosis.

    Example of implementation: https://tntorch.readthedocs.io/en/latest/_modules/metrics.html
    """

    def __init__(self, device, alpha=1.0, eps=1e-8):
        super().__init__(device, alpha, eps)

    def __call__(self, x):
        # Find kurtosis for each feature
        std_x = torch.maximum(torch.std(x, dim=0), self.eps)
        dev = x - x.mean(axis=0)

        kurtosises = torch.mean((dev / std_x) ** 4, dim=0) - 3
        kurtosis = torch.abs(kurtosises).mean()

        return self.alpha / max(kurtosis, self.eps)


class CorrMatLoss(BaseIndependenceLoss):
    def __init__(self, device, alpha=10.0, eps=1e-8):
        super().__init__(device, alpha, eps)

    def __call__(self, x):
        return self.alpha * non_diagonal_correlation(x)
