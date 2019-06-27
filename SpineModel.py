import torchvision.models as models
import torch
import torch.nn as nn

"""
def spine_model():
    # No avgpool and fc
    image_layers = list(models.resnet18().children())[:-2]
    # Rewrite 1st layer to support gray scale image
    image_layers[0] = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    conv = nn.Conv2d(512, 4, (7, 7), padding=(3, 3))
    image_layers.append(conv)
    image_model = nn.Sequential(*image_layers)
    print(list(image_model.children()))
    return image_model
"""


class SpineModel(nn.Module):
    def __init__(self):
        super(SpineModel, self).__init__()
        # No avgpool and fc
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        rn = models.resnet18()
        self.bn1 = rn.bn1
        self.relu = rn.relu
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.conv2 = nn.Conv2d(256, 256, (3, 3), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 4, (7, 7), padding=(3, 3))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)  # Stride 2
        x = self.layer3(x)  # Stride 2

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        return x
