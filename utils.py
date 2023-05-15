from datasets.cifar10 import CIFAR10
from datasets.oxford5k import Oxford5k
from datasets.google_landmarks import GoogleLandmarks
from losses.zero_loss import ZeroLoss
from losses.independence import NegApproxLoss1, NegApproxLoss2, KurtosisLoss, CorrMatLoss
from losses.regularization import L1
from whitening import ZCA1, ZCA2, NoWhitening


def get_dataset(dataset_name, *args):
    if dataset_name == 'CIFAR10':
        return CIFAR10(*args)
    elif dataset_name == 'Oxford5k':
        return Oxford5k(*args)
    elif dataset_name == 'GoogleLandmarks':
        return GoogleLandmarks(*args)
    else:
        raise Exception(f"Unknown dataset {dataset_name}")


def get_independence_loss(independence_loss, device):
    if independence_loss is None:
        return ZeroLoss(device)
    elif independence_loss == 'NegApproxLoss1':
        return NegApproxLoss1(device)
    elif independence_loss == 'NegApproxLoss2':
        return NegApproxLoss2(device)
    elif independence_loss == 'KurtosisLoss':
        return KurtosisLoss(device)
    elif independence_loss == 'CorrMatLoss':
        return CorrMatLoss(device)
    else:
        raise Exception(f"Unknown regularization loss {independence_loss}")


def get_regularization_loss(regularization_loss, device):
    if regularization_loss is None:
        return ZeroLoss(device)
    elif regularization_loss == 'L1':
        return L1()
    else:
        raise Exception(f"Unknown regularization loss {regularization_loss}")


def get_whitening(whitening, device, embedding_dim):
    if whitening == 'ZCA1':
        return ZCA1(device=device, num_features=embedding_dim)
    elif whitening == 'ZCA2':
        return ZCA2(device=device, num_features=embedding_dim, num_groups=1, T=5, dim=2, affine=False)
    else:
        return NoWhitening()
