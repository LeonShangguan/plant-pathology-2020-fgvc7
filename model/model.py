import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from pytorchcv.model_provider import get_model as ptcv_get_model
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class PlantModel(nn.Module):

    def __init__(self, model_name="resnet34", num_classes=4):
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        if model_name.lower() == 'resnet34':

            self.backbone = ptcv_get_model("resnet34", pretrained=True)
            self.feature_size = 512
            self.backbone.features.final_pool = Identity()

        elif model_name.lower() == 'efficientnet_b7':

            self.backbone = EfficientNet.from_pretrained('efficientnet-b7')
            self.feature_size = 2560

        elif model_name.lower() == 'se_resnext101':

            self.backbone = ptcv_get_model("seresnext101_32x4d", pretrained=True)
            self.feature_size = 2048
            self.backbone.features.final_pool = Identity()

        elif model_name.lower() == "se_resnext50":

            self.backbone = ptcv_get_model("seresnext50_32x4d", pretrained=True)
            self.feature_size = 2048
            self.backbone.features.final_pool = Identity()

        elif model_name.lower() == 'inceptionresnetv2':

            self.backbone = ptcv_get_model("inceptionresnetv2", pretrained=True)
            self.feature_size = 1536
            self.backbone.features.final_pool = Identity()

        elif model_name.lower() == 'pnasnet5large':

            self.backbone = ptcv_get_model("pnasnet5large", pretrained=True)
            self.feature_size = 4320
            self.backbone.features.final_pool = Identity()

        else:
            raise NotImplementedError

        self.avg_poolings = AdaptiveConcatPool2d()
        self.tail = nn.Sequential(nn.Linear(self.feature_size * 2, 512), Mish())
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.fc = nn.Linear(512, self.num_classes)

    def get_logits_by_random_dropout(self, fuse_hidden, fc):

        logit = None
        h = fuse_hidden

        for j, dropout in enumerate(self.dropouts):

            if j == 0:
                logit = fc(dropout(h))
            else:
                logit += fc(dropout(h))

        return logit / len(self.dropouts)

    def forward(self, x):

        bs = x.shape[0]

        if "efficientnet" in self.model_name.lower():
            x = self.backbone.extract_features(x)
        else:
            x = self.backbone.features(x)

        logit = self.avg_poolings(x)
        logit = logit.view(bs, -1)

        logit = self.tail(logit)
        logit = self.get_logits_by_random_dropout(logit, self.fc)
    
        return logit


def test_Net():
    print("------------------------testing Net----------------------")

    x = torch.tensor(np.random.random((8, 3, 512, 512)).astype(np.float32)).cuda()
    model = PlantModel().cuda()
    logits = model(x)
    print(logits.shape)
    print("------------------------testing Net finished----------------------")

    return


if __name__ == '__main__':
    model = PlantModel()
    # print(model)
    test_Net()