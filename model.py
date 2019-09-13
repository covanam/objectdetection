import time
import torch.nn as nn
import torch.nn.functional as func
import torch


def conv(din, dout, kernel, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(din, dout, kernel, stride, padding, bias=False),
        nn.BatchNorm2d(dout),
        nn.ReLU(inplace=True)
    )


class ConvRes(nn.Module):
    def __init__(self, din):
        super().__init__()
        self.layers = nn.Sequential(
            conv(din, din, 3, 1, 1),
            conv(din, din, 3, 1, 1)
        )

    def forward(self, x):
        return x + self.layers(x)


class ConvPool(nn.Module):
    def __init__(self, din, dout):
        super().__init__()
        self.layers = conv(din, dout, 2, 2)

    def forward(self, x):
        return self.layers(x)


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

        # input: 512x512
        self.first_layer = conv(3, 16, 3, 1, 1)

        self.image_transform = nn.Sequential(
            # 512x512
            ConvRes(16),
            ConvPool(16, 32),
            # 256x256
            ConvRes(32),
            ConvPool(32, 64),
            # 128x128  /  16x16
        )

        self.feature_extractor = nn.Sequential(
            ConvRes(64),
            ConvRes(64),
            ConvPool(64, 128),
            # 8x8
            ConvRes(128),
            ConvRes(128)
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Conv2d(128, 256, 8, 4, bias=True),
            torch.nn.Tanh(),
            torch.nn.Conv2d(256, 50, 1, 1, bias=True)
        )

    def forward(self, x):
        # basic transforms
        x = self.first_layer(x)
        x = self.image_transform(x)

        # split into 4 scale
        x1 = x  # 128x128
        x2 = func.avg_pool2d(x1, 2)  # 64x64
        x3 = func.avg_pool2d(x2, 2)  # 32x32
        x4 = func.avg_pool2d(x3, 2)  # 16x16

        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        x3 = self.feature_extractor(x3)
        x4 = self.feature_extractor(x4)

        x1 = self.classifier(x1)
        x2 = self.classifier(x2)
        x3 = self.classifier(x3)
        x4 = self.classifier(x4)

        return x1, x2, x3, x4


if __name__ == '__main__':
    net = MyNet().eval()
    torch.save(net.state_dict(), 'model/model')
    im = torch.zeros((1, 3, 512, 512))
    with torch.no_grad():
        start = time.time()
        pred = net(im)
        stop = time.time()
        print(stop - start)
    for p in pred:
        print(p.shape)
