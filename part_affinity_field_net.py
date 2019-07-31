import torchvision.models as models
import torch
import torch.nn as nn

class SpineModelPAF(nn.Module):
    def __init__(self):
        super(SpineModelPAF, self).__init__()
        self.pcm_n = 6
        self.paf_n = 1

        import torchvision.models.vgg as vgg
        vgg19 = vgg.vgg19_bn(pretrained=False)
        top_layers = list(list(vgg19.children())[0].children())
        top_layers[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        tops = top_layers[:33]  # Top 10 (conv batch relu)*10 + maxpool * 3
        tops.pop(26)  # delete third max pool
        [tops.append(l) for l in self.make_conv_layers(512, 256)]
        [tops.append(l) for l in self.make_conv_layers(256, 128)]
        self.model_0 = nn.Sequential(*tops)  # out: 32, 94

        s1_pcm = lambda: self.stage1(self.pcm_n)
        s1_paf = lambda: self.stage1(self.paf_n)
        sn_pcm = lambda: self.stageN(self.pcm_n)
        sn_paf = lambda: self.stageN(self.paf_n)

        self.model1_1 = s1_pcm()
        self.model1_2 = s1_paf()

        self.model2_1 = sn_pcm()
        self.model2_2 = sn_paf()

        self.model3_1 = sn_pcm()
        self.model3_2 = sn_paf()

        self.model4_1 = sn_pcm()
        self.model4_2 = sn_paf()

        self.model5_1 = sn_pcm()
        self.model5_2 = sn_paf()

    def make_conv_layers(self, in_channels, out_channels, kernels=3, padding=1, ReLU=True):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernels, padding=padding)
        if ReLU:
            layers = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        else:
            layers = [conv2d]
        return nn.Sequential(*layers)

    def stage1(self, out_channels):
        layers = []
        layers.append(self.make_conv_layers(128, 128))
        layers.append(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1))
        layers.append(self.make_conv_layers(64, 64))
        layers.append(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1))
        layers.append(self.make_conv_layers(32, 32))
        layers.append(self.make_conv_layers(32, 32))
        layers.append(self.make_conv_layers(32, 32, kernels=1, padding=0))
        layers.append(self.make_conv_layers(32, out_channels, kernels=1, padding=0, ReLU=False))
        return nn.Sequential(*layers)

    def stageN(self, out_channels):
        layers = []
        layers.append(self.make_conv_layers(128+self.pcm_n+self.paf_n, 128, kernels=7, padding=3))
        layers.append(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1))
        layers.append(self.make_conv_layers(64, 64, kernels=7, padding=3))
        layers.append(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1))
        layers.append(self.make_conv_layers(32, 32, kernels=7, padding=3))
        layers.append(self.make_conv_layers(32, 32, kernels=7, padding=3))
        layers.append(self.make_conv_layers(32, 32, kernels=1, padding=0))
        layers.append(self.make_conv_layers(32, out_channels, kernels=1, padding=0, ReLU=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.model_0(x)
        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)

        out2 = torch.cat([out1_1, out1_2, out1], dim=1)
        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)

        out3 = torch.cat([out2_1, out2_2, out1], dim=1)
        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)

        out4 = torch.cat([out3_1, out3_2, out1], dim=1)
        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)

        out5 = torch.cat([out4_1, out4_2, out1], dim=1)
        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)

        loss1_pcm_img = torch.stack([out1_1, out2_1, out3_1, out4_1, out5_1])
        loss2_paf_img = torch.stack([out1_2, out2_2, out3_2, out4_2, out5_2])

        loss1_pcm_img = torch.mean(loss1_pcm_img, dim=0)
        loss2_paf_img = torch.mean(loss2_paf_img, dim=0)

        return out5_1, out5_2, loss1_pcm_img, loss2_paf_img


class CobbAngleModel(nn.Module):
    """
    This model is used to predict cobb angles from heatmaps
    """

    def __init__(self, in_channels=1):

        super(CobbAngleModel, self).__init__()

        cfgs = [256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 512]

        def make_conv_layers(in_channels, out_channels, kernels=3, padding=1, ReLU=True):
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernels, padding=padding)
            if ReLU:
                layer = nn.Sequential(conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))
            else:
                layer = nn.Sequential(conv2d)
            return layer

        self.conv_layers = nn.Sequential()

        for i, cfg in enumerate(cfgs):
            if cfg == 'M':
                layer = nn.MaxPool2d(kernel_size=2, stride=2)
            else:
                layer = make_conv_layers(in_channels, cfg)
                in_channels = cfg
            self.conv_layers.add_module("cfg{}".format(i), layer)

        self.conv_layers.add_module("avg", nn.AdaptiveAvgPool2d((4,4)))
        # Dense layers
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 2048),
            nn.Dropout(),
            nn.ReLU(True),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 3),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



