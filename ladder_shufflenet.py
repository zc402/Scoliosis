import torch
import torch.nn as nn

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class InvertedResidualUpsample(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidualUpsample, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv_upsample(inp, inp, kernel_size=4, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

            self.branch2 = nn.Sequential(
                nn.Conv2d(inp if (self.stride > 1) else branch_features,
                          branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                self.depthwise_conv_upsample(branch_features, branch_features, kernel_size=4, stride=self.stride, padding=1),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:  # stride == 1
            self.branch2 = nn.Sequential(
                nn.Conv2d(inp if (self.stride > 1) else branch_features,
                          branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
                self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride,
                                             padding=1),
                nn.BatchNorm2d(branch_features),
                nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )

    @staticmethod
    def depthwise_conv_upsample(i, o, kernel_size, stride, padding, bias=False):
        return nn.ConvTranspose2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class LadderModel(nn.Module):

    def __init__(self, out_channels=3):
        super(LadderModel, self).__init__()

        self._stage_out_channels = [48, 48, 96, 192, 1024]  # init, e1, e2, e3, e4

        self._stage_in_channels_dec = [1024, 192+192, 96+96, 48+48, 48]  # in: d4, d3, d2, d1, final
        self._stage_out_channels_dec = [192, 96, 48, 48, out_channels]  # out: d4, d3, d2, d1, final

        self.initial = nn.Sequential(
            nn.Conv2d(1, self._stage_out_channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self._stage_out_channels[0]),
            nn.ReLU(inplace=True),
        )

        # Encoder
        input_channels = self._stage_out_channels[0]
        stage_names = ['encoder{}'.format(i) for i in [1, 2, 3, 4]]
        stages_repeats = [8, 16, 8, 8]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [InvertedResidual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        # Decoder
        stage_names = ['decoder{}'.format(i) for i in [4, 3, 2, 1]]
        stages_repeats = [8, 8, 16, 8]
        for name, repeats, input_channels, output_channels in zip(
                stage_names, stages_repeats,self._stage_in_channels_dec[:4], self._stage_out_channels_dec[:4]):
            seq = [InvertedResidualUpsample(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(InvertedResidualUpsample(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))

        # Final Block
        input_channels = self._stage_in_channels_dec[-1]
        output_channels = self._stage_out_channels_dec[-1]
        self.final = nn.Sequential(
            nn.Conv2d(input_channels, input_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(input_channels, output_channels, 1, 1, 0)
        )

    def forward(self, x):
        init = self.initial(x)
        e1 = self.encoder1(init)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4)
        d4_cat = torch.cat((d4, e3), dim=1)
        d3 = self.decoder3(d4_cat)
        d3_cat = torch.cat((d3, e2), dim=1)
        d2 = self.decoder2(d3_cat)
        d2_cat = torch.cat((d2, e1), dim=1)
        d1 = self.decoder1(d2_cat)

        final = self.final(d1)

        return final[:, 0:2, :, :], final[:, 2:3, :, :]  # pcm, paf

if __name__=="__main__":
    import numpy as np
    ladder = LadderModel().cuda()
    print(ladder)
    input = np.zeros([2, 1, 256, 256], np.float32)
    t_input = torch.from_numpy(input).cuda()
    out = ladder(t_input)
