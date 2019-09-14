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

        self.first_layer = conv(3, 16, 3, 1, 1)  # 256 256

        self.tf1 = ConvRes(16)  # 256
        self.rs1 = ConvPool(16, 16) # 128
        self.tf2 = ConvRes(16)  # 128
        self.rs2 = ConvPool(16, 16) # 64

        self.fe = nn.Sequential(
            ConvRes(16),
            ConvPool(16, 32),
            ConvRes(32),
            ConvRes(32),
            ConvPool(32, 64),
            ConvRes(64),
            ConvRes(64),
            ConvPool(64, 128),
            ConvRes(128),
            ConvRes(128),
            ConvRes(128),
            ConvRes(128)
        )
    
        self.classifier = torch.nn.Sequential(
            nn.Conv2d(128, 256, 8, 4, bias=True),
            nn.Tanh(),
            nn.Conv2d(256, 50, 1, 1, bias=True)
        )

    def forward(self, x):
        # basic transforms
        x = self.first_layer(x)

        # split into 4 scale
        x1 = self.tf1(x)
        x2 = self.rs1(x1)

        x1 = self.tf2(x1)
        x2 = self.tf2(x2)
        x3 = self.rs2(x2)
        
        x1 = self.fe(x1)
        x2 = self.fe(x2)
        x3 = self.fe(x3)

        x1 = self.classifier(x1)
        x2 = self.classifier(x2)
        x3 = self.classifier(x3)

        return x3, x1, x2, x3

if __name__ == '__main__':
    net = MyNet().eval()
    torch.save(net.state_dict(), 'model/model')
    im = torch.zeros((1, 3, 256, 256))
    with torch.no_grad():
        start = time.time()
        pred = net(im)
        stop = time.time()
        print(stop - start)
    for p in pred:
        print(p.shape)
