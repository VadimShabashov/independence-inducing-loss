import torch

from whitening import ZCA1, ZCA2, NoWhitening


class BaseIndependenceLoss:
    """
    Base class for independence loss.
    """
    def __init__(self, device, whitening, embedding_dim, alpha, eps):
        self.alpha = alpha
        self.eps = torch.tensor(eps, device=device)
        self.device = device

        # Set whitening approach
        if whitening == 'ZCA1':
            self.whitening = ZCA1(device=device, num_features=embedding_dim)
        elif whitening == 'ZCA2':
            self.whitening = ZCA2(
                device=device, num_features=embedding_dim, num_groups=1, T=5, dim=2, affine=False
            )
        else:
            self.whitening = NoWhitening()

    def apply_whitening(self, x):
        return self.whitening(x)


class NegApproxLoss1(BaseIndependenceLoss):
    """
    Minimization of mutual information using negentropy approximation from:
    https://www.cs.helsinki.fi/u/ahyvarin/papers/NIPS97.pdf

    Minimization of mutual information <=> maximization of negentropy <=> minimization of 1/negentropy
    """

    def __init__(self, device, whitening, embedding_dim, alpha=10.0, eps=1e-8):
        super().__init__(device, whitening, embedding_dim, alpha, eps)

    def __call__(self, x):
        # Apply whitening
        x = self.apply_whitening(x)

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

    def __init__(self, device, whitening, embedding_dim, alpha=10.0, eps=1e-8):
        super().__init__(device, whitening, embedding_dim, alpha, eps)

    def __call__(self, x):
        # Apply whitening
        x = self.apply_whitening(x)

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

    def __init__(self, device, whitening, embedding_dim, alpha=1.0, eps=1e-8):
        super().__init__(device, whitening, embedding_dim, alpha, eps)

    def __call__(self, x):
        # Apply whitening
        x = self.apply_whitening(x)

        # Find kurtosis for each feature
        std_x = torch.maximum(torch.std(x, dim=0), self.eps)
        dev = x - x.mean(axis=0)

        kurtosises = torch.mean((dev / std_x) ** 4, dim=0) - 3
        kurtosis = torch.abs(kurtosises).mean()

        return self.alpha / max(kurtosis, self.eps)


class CorrMatLoss(BaseIndependenceLoss):
    def __init__(self, device, whitening, embedding_dim, alpha=10.0, eps=1e-8):
        super().__init__(device, whitening, embedding_dim, alpha, eps)

    def __call__(self, x):
        # Apply whitening
        x = self.apply_whitening(x)

        # Calculate correlation matrix
        correlation_matrix = torch.corrcoef(x.T)

        # Create mask for diagonal elements
        diagonal_elements_mask = torch.eye(correlation_matrix.shape[0], dtype=torch.bool)

        # Get non-diagonal elements
        non_diagonal_elements = correlation_matrix[~diagonal_elements_mask]

        return self.alpha * non_diagonal_elements.abs().mean()
