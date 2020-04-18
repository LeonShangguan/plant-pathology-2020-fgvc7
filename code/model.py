from library import *
from config import Config


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

    def __init__(self, num_classes=4):
        super().__init__()
        if Config.model_name.lower() == 'resnet34':
            self.backbone = ptcv_get_model("resnet34", pretrained=True)

            self.backbone.features.final_pool = nn.AdaptiveAvgPool2d(1)
            self.backbone.output = nn.Linear(512, num_classes)
            # self.backbone.output = nn.Sequential(nn.Linear(512, 128),
            #                                      swish(),
            #                                      nn.Dropout(p=0.5),
            #                                      nn.Linear(128, num_classes))
        elif Config.model_name.lower() == 'efficientnet_b7':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b7')
            # self.backbone._fc = nn.Linear(2560, num_classes)

            self.backbone._fc = nn.Sequential(nn.Linear(2560, 256),
                                              Mish(),
                                              nn.Dropout(p=0.5),
                                              nn.Linear(256, num_classes))
        elif Config.model_name.lower() == 'se_resnext101':
            self.backbone = ptcv_get_model("seresnext101_32x4d", pretrained=True)

            self.backbone.features.final_pool = nn.AdaptiveAvgPool2d(1)
            self.backbone.output = nn.Sequential(nn.Linear(2048, 256),
                                                 Mish(),
                                                 nn.Dropout(p=0.5),
                                                 nn.Linear(256, num_classes))
        elif Config.model_name.lower() == 'inceptionresnetv2':
            self.backbone = ptcv_get_model("inceptionresnetv2", pretrained=True)

            self.backbone.features.final_pool = nn.AdaptiveAvgPool2d(1)
            self.backbone.output = nn.Sequential(nn.Linear(1536, 128),
                                                 Mish(),
                                                 nn.Dropout(p=0.5),
                                                 nn.Linear(128, num_classes))
        elif Config.model_name.lower() == 'pnasnet5large':
            self.backbone = ptcv_get_model("pnasnet5large", pretrained=True)
            # self.backbone.features.final_pool = Identity()
            #
            # self.output = nn.Sequential(Mish(),
            #                             Conv2dBN(4320, 128, kernel_size=1),
            #                             GeM(),
            #                             # nn.AdaptiveAvgPool2d(1),
            #                             nn.Linear(3072, 128),
            #                             Mish(),
            #                             nn.Dropout(p=0.5),
            #                             nn.Linear(128, num_classes))

            self.backbone.features.final_pool = nn.AdaptiveAvgPool2d(1)
            self.backbone.output = nn.Sequential(nn.Linear(4320, 512),
                                                 Mish(),
                                                 nn.Dropout(p=0.5),
                                                 nn.Linear(512, num_classes))
        else:
            raise NotImplementedError

    def forward(self, x):

        x = self.backbone(x)
        x = F.softmax(x)

        return x


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
    print(model)
    test_Net()
