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
        import torchvision.models.resnet as resnet
        # No avgpool and fc
        heat_number = 7
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3, bias=False)
        rn = models.resnet18(pretrained=True)
        self.bn1 = rn.bn1
        self.relu1 = rn.relu
        self.layer1 = rn.layer1
        self.layer2 = rn.layer2
        self.layer3 = rn.layer3
        self.conv2 = nn.Conv2d(256, 256, (3, 3), padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(256, heat_number, (7, 7), padding=(3, 3))

        # Predict position of each point
        rn.inplanes = 256+heat_number
        self.layer4 = rn._make_layer(resnet.BasicBlock, 512, 2, stride=2)  # 94, 32
        rn.inplanes = 512
        self.layer5 = rn._make_layer(resnet.BasicBlock, 512, 2, stride=2)  # 47, 16
        self.avgpool = rn.avgpool
        self.fc = nn.Linear(512, 68*2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)  # Stride 2
        x_div4 = self.layer3(x)  # Stride 2

        x = self.conv2(x_div4)
        x = self.bn2(x)
        x = self.relu2(x)
        gaussians = self.conv3(x)

        # Points
        y = torch.cat((gaussians, x_div4), dim=1)
        y = self.layer4(y)
        y = self.layer5(y)
        y = self.avgpool(y)
        y = y.reshape(y.size(0), -1)
        pts = self.fc(y)

        return gaussians, pts
