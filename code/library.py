import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder

from pytorchcv.model_provider import get_model as ptcv_get_model
from efficientnet_pytorch import EfficientNet


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import torchvision
import torch.optim as optim

import albumentations as A
from albumentations.pytorch import ToTensorV2




class DenseCrossEntropy(nn.Module):

    def __init__(self):
        super(DenseCrossEntropy, self).__init__()

    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()

        logprobs = F.log_softmax(logits, dim=-1)

        loss = -labels * logprobs
        loss = loss.sum(-1)

        return loss.mean()


def seed_everything(seed):
    np.random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def cal_score(labels, predictions):
    healthy_true = labels[:, 0]
    multiple_diseases_true = labels[:, 1]
    rust_true = labels[:, 2]
    scab_true = labels[:, 3]

    healthy_pre = predictions[:, 0]
    multiple_diseases_pre = predictions[:, 1]
    rust_pre = predictions[:, 2]
    scab_pre = predictions[:, 3]

    healthy_auc = roc_auc_score(healthy_true, healthy_pre)
    multiple_diseases_auc = roc_auc_score(multiple_diseases_true, multiple_diseases_pre)
    rust_auc = roc_auc_score(rust_true, rust_pre)
    scab_auc = roc_auc_score(scab_true, scab_pre)

    avg_auc = (healthy_auc+multiple_diseases_auc+rust_auc+scab_auc)/4.0

    return np.asarray([avg_auc, healthy_auc, multiple_diseases_auc, rust_auc, scab_auc])



