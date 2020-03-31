from library import *
from config import Config


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class PlantModel(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()
        if Config.model_name.lower() == 'resnet34':
            self.backbone = ptcv_get_model("resnet34", pretrained=True)

            self.backbone.features.final_pool = nn.AdaptiveAvgPool2d(1)
            self.backbone.output = nn.Linear(512, num_classes)
        elif Config.model_name.lower() == 'efficientnet_b7':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')

            self.backbone._fc = nn.Linear(1280, num_classes)
        elif Config.model_name.lower() == 'se_resnext101':
            self.backbone = ptcv_get_model("seresnext101_32x4d", pretrained=True)

            self.backbone.features.final_pool = nn.AdaptiveAvgPool2d(1)
            self.backbone.output = nn.Linear(2048, num_classes)
        else:
            raise NotImplementedError

    def forward(self, x):

        x = self.backbone(x)

        return x


def test_Net():
    print("------------------------testing Net----------------------")

    x = torch.tensor(np.random.random((8, 3, 512, 512)).astype(np.float32)).cuda()
    model = PlantModel().cuda()
    print(model.state_dict().keys())
    logits = model(x)
    print(logits.shape)
    print(logits[0], logits[1], logits[2], logits[3])
    print("------------------------testing Net finished----------------------")

    return


if __name__ == '__main__':
    model = PlantModel()
    print(model)
    test_Net()
