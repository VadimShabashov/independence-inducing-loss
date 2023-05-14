from datasets.cifar10 import CIFAR10
from datasets.oxford5k import Oxford5k
from datasets.google_landmarks import GoogleLandmarks
from losses.zero_loss import ZeroLoss
from losses.independence import NegApproxLoss1, NegApproxLoss2, KurtosisLoss, CorrMatLoss
from losses.regularization import L1


def get_dataset(dataset_name, *args):
    if dataset_name == 'CIFAR10':
        return CIFAR10(*args)
    elif dataset_name == 'Oxford5k':
        return Oxford5k(*args)
    elif dataset_name == 'GoogleLandmarks':
        return GoogleLandmarks(*args)
    else:
        raise Exception(f"Unknown dataset {dataset_name}")


def get_independence_loss(independence_loss, device, whitening, embedding_dim):
    if independence_loss is None:
        return ZeroLoss(device)
    elif independence_loss == 'NegApproxLoss1':
        return NegApproxLoss1(device, whitening, embedding_dim)
    elif independence_loss == 'NegApproxLoss2':
        return NegApproxLoss2(device, whitening, embedding_dim)
    elif independence_loss == 'KurtosisLoss':
        return KurtosisLoss(device, whitening, embedding_dim)
    elif independence_loss == 'CorrMatLoss':
        return CorrMatLoss(device, whitening, embedding_dim)
    else:
        raise Exception(f"Unknown regularization loss {independence_loss}")


def get_regularization_loss(regularization_loss, device):
    if regularization_loss is None:
        return ZeroLoss(device)
    elif regularization_loss == 'L1':
        return L1()
    else:
        raise Exception(f"Unknown regularization loss {regularization_loss}")
