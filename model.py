import time
import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvBlock(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(din, din, 3, 1, 1, bias=False),
            nn.BatchNorm2d(din),
            nn.ReLU(inplace=True),
            nn.Conv2d(din, din, 3, 1, 1, bias=False),
            nn.BatchNorm2d(din),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.Sequential(
            nn.Conv2d(din, dout, 2, 2, 0, bias=False),
            nn.BatchNorm2d(dout),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = x + self.conv(x)
        return self.pool(x)


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class MobileBottleneck(nn.Module):
    def __init__(self, inp, exp, oup, kernel, stride):
        super(MobileBottleneck, self).__init__()
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        nlin_layer = Hswish

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d

        self.layers = nn.Sequential(
            # pw
            conv_layer(128, 768, 1, 1, 0, bias=False),
            norm_layer(768),
            Hswish(inplace=True),
            # dw
            conv_layer(768, 768, 4, 2, 0, groups=768, bias=False),
            norm_layer(768),
            nn.Tanh(),
            # pw-linear
            conv_layer(768, 50, 1, 1, 0, bias=False),
        )

    def forward(self, x):
        return self.layers(x)


class MBBlock(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.conv = nn.Sequential(
            MobileBottleneck(din, 6 * din, din, 3, 1),
            MobileBottleneck(din, 6 * din, din, 3, 1),
            MobileBottleneck(din, 6 * dout, dout, 2, 2)
        )

    def forward(self, x):
        return self.conv(x)


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )

        self.block1 = nn.Sequential(
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )  # 32

        self.block2 = MBBlock(128, 128)  # 16
        self.block3 = MBBlock(128, 128)  # 8
        self.block4 = MBBlock(128, 128)  # 4

        self.decoder1 = Decoder()
        self.decoder2 = Decoder()
        self.decoder3 = Decoder()
        self.decoder4 = Decoder()

    def forward(self, x):
        x = self.first_layer(x)

        x1 = self.block1(x)  # 32x32
        detection1 = self.decoder1(x1)

        x2 = self.block2(x1)  # 16x16
        detection2 = self.decoder2(x2)

        x3 = self.block3(x2)
        detection3 = self.decoder3(x3)

        x4 = self.block4(x3)
        detection4 = self.decoder4(x4)

        return detection1, detection2, detection3, detection4


if __name__ == '__main__':
    net = MyNet().eval()
    torch.save(net.state_dict(), 'model/model')
    im = torch.zeros((1, 3, 256, 256))
    with torch.no_grad():
        pred = net(im)
    for p in pred:
        print(p.shape)
