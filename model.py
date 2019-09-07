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


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d

        self.layers = nn.Sequential(
            conv_layer(256, 512, 1, 1, 0, bias=False),
            norm_layer(512),
            nn.ReLU(True),
            
            conv_layer(512, 512, 4, 2, 0, bias=False),
            norm_layer(512),
            nn.Tanh(),
            
            conv_layer(512, 50, 1, 1, 0, bias=True),
        )

    def forward(self, x):
        return self.layers(x)


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.first_layer = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

        self.block1 = nn.Sequential(
            ConvBlock(32, 64),  # 64x128x128
            ConvBlock(64, 128),  # 128x64x64
            ConvBlock(128, 256),  # 256x32x32
        )  # 32

        self.block2 = ConvBlock(256, 256)  # 16
        self.block3 = ConvBlock(256, 256)  # 8
        self.block4 = ConvBlock(256, 256)  # 4


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
    net = MyNet().eval().cuda()
    torch.save(net.state_dict(), 'model/model')
    im = torch.zeros((1, 3, 256, 256)).cuda()
    with torch.no_grad():
        pred=net(im)
        start = time.time()
        pred = net(im)
        pred = net(im)
        pred = net(im)
        pred = net(im)
        pred = net(im)
        stop = time.time()
        print(stop - start)
    for p in pred:
        print(p.shape)
