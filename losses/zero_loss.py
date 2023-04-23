import torch


class ZeroLoss:
    def __init__(self, device):
        self.device = device

    def __call__(self, *args):
        return torch.tensor(0.0, device=self.device)
