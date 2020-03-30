import os
import cv2
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.metrics import roc_auc_score
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
