from library import *
from config import Config


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class PlantModel(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        if Config.model_name.lower() == 'resnet34':
            self.backbone = ptcv_get_model("resnet34", pretrained=True)

            self.backbone.features.final_pool = nn.AdaptiveAvgPool2d(1)
            self.backbone.output = nn.Sequential(nn.Dropout(0.5), nn.Linear(512, num_classes))
        elif Config.model_name.lower() == 'se_resnext101':
            self.backbone = ptcv_get_model("seresnext101_32x4d", pretrained=True)

            self.backbone.features.final_pool = nn.AdaptiveAvgPool2d(1)
            self.backbone.output = nn.Sequential(nn.Linear(2048, 256),
                                                 # Mish(),
                                                 nn.Dropout(p=0.5),
                                                 nn.Linear(256, num_classes))
        elif Config.model_name.lower() == 'efficientnet_b7':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b7')
            # self.backbone._fc = nn.Linear(2560, num_classes)

            self.backbone._fc = nn.Sequential(nn.Linear(2560, 256),
                                              Mish(),
                                              nn.Dropout(p=0.5),
                                              nn.Linear(256, num_classes))
        else:
            self.backbone = torchvision.models.resnet34(pretrained=True)
            in_features = self.backbone.fc.in_features
            self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        if Config.model_name.lower() == 'resnet34':
            x = self.backbone(x)
        elif Config.model_name.lower() == 'se_resnext101':
            x = self.backbone(x)
        elif Config.model_name.lower() == 'efficientnet_b7':
            x = self.backbone(x)
        else:
            batch_size, C, H, W = x.shape

            x = self.backbone.conv1(x)
            x = self.backbone.bn1(x)
            x = self.backbone.relu(x)
            x = self.backbone.maxpool(x)

            x = self.backbone.layer1(x)
            x = self.backbone.layer2(x)
            x = self.backbone.layer3(x)
            x = self.backbone.layer4(x)

            x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
            x = F.dropout(x, 0.25, self.training)

            x = self.logit(x)

        return x