import torch
import torch.nn as nn
import torch.nn.functional as func
import mnasnet

vgg = torch.nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1, stride=1, bias=False),
    nn.LeakyReLU(negative_slope=0.05, inplace=True),
    nn.AvgPool2d(2),

    nn.Conv2d(32, 64, 3, padding=1, stride=1, bias=False),
    nn.LeakyReLU(negative_slope=0.05, inplace=True),
    nn.AvgPool2d(2),

    nn.Conv2d(64, 128, 3, padding=1, stride=1, bias=False),
    nn.LeakyReLU(negative_slope=0.05, inplace=True),
    nn.AvgPool2d(2),

    nn.Conv2d(128, 1280, 3, padding=1, stride=1, bias=False),
    nn.LeakyReLU(negative_slope=0.05, inplace=True),
    nn.AvgPool2d(2)
)


class MyNet(nn.Module):
    @staticmethod
    def _detector():
        return nn.Sequential(
            nn.Conv2d(1280, 2409, 1, padding=0, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(2409, 2409, 1, padding=0, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(2409, 14, 1, padding=0, stride=1, bias=True)
        )

    def __init__(self, base_network=vgg):
        super().__init__()
        self.feature_extractor = base_network
       
        self.detector1 = self._detector()
        self.detector2 = self._detector()
        self.detector3 = self._detector()
        self.detector4 = self._detector()

    def forward(self, x):
        x = self.feature_extractor(x)  # 16x16

        fm1 = func.avg_pool2d(x, 2, stride=1)
        detection1 = self.detector1(fm1)  # 15x15

        fm2 = func.avg_pool2d(x, 4, stride=2)
        detection2 = self.detector2(fm2)  # 7x7

        fm3 = func.avg_pool2d(x, 8, stride=4)
        detection3 = self.detector3(fm3)  # 3x3

        fm4 = func.avg_pool2d(x, 16, stride=8)
        detection4 = self.detector4(fm4)

        return detection1, detection2, detection3, detection4

def _forward(self, x):
        x = self.layers(x)

def model():
    model = MyNet()
    del model.classifier
    model.load_state_dict(torch.load('model/model'))
    model.feature_extractor.forward = _forward
    return model

if __name__ == '__main__':
    model = MyNet()
    model.load_state_dict(torch.load('model/pretrained'))
    del model.classifier
    model.forward = _forward
    torch.save(model.state_dict(), 'model/model')
