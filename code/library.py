import os
import cv2
import time
import math
import numpy as np
import pandas as pd
from apex import amp
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder
import logging
from pytorchcv.model_provider import get_model as ptcv_get_model
from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

import albumentations as A
from albumentations.pytorch import ToTensorV2

from os.path import isfile
import torch.nn.init as init
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, OneHotEncoder
from torch.utils.data import Dataset
from torchvision import transforms
import time
import math
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from sklearn import metrics
import urllib
import pickle
import cv2
import torch.nn.functional as F
from torchvision import models
import seaborn as sns
import random
import sys
import shutil
import albumentations
from albumentations import pytorch as AT
from pytorchcv.model_provider import get_model as ptcv_get_model


from apex import amp
from efficientnet_pytorch import EfficientNet
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from torchcontrib.optim import SWA

# torch.backends.cudnn.benchmark = True












