import torch
import torch.nn as nn
import torch.nn.functional as func
import mobilenetv3 as mb3


class MyNet(nn.Module):
    @staticmethod
    def _detector():
        return nn.Sequential(
            nn.Conv2d(160, 960, 1, padding=0, stride=1, bias=True),
            nn.Dropout2d(inplace=True, p=0),
            mb3.Hswish(True),
            
            nn.Conv2d(960, 1280, 1, padding=0, stride=1, bias=True),
            nn.Dropout2d(inplace=True, p=0),
            mb3.Hswish(True),
            
            nn.Conv2d(1280, 14, 1, padding=0, stride=1, bias=True)
        )

    def __init__(self, base_network=mb3.MobileNetV3(mode='large', width_mult=1.0)):
        super().__init__()
        self.feature_extractor = base_network
       
        self.detector1 = self._detector()
        self.detector2 = self._detector()
        self.detector3 = self._detector()
        self.detector4 = self._detector()

    def forward(self, x):
        x = self.feature_extractor(x)
        
        fm1 = func.avg_pool2d(x, 2, stride=1)
        detection1 = self.detector1(fm1)  # 15x15

        fm2 = func.avg_pool2d(x, 4, stride=2)
        detection2 = self.detector2(fm2)  # 7x7

        fm3 = func.avg_pool2d(x, 8, stride=4)
        detection3 = self.detector3(fm3)  # 3x3

        fm4 = func.avg_pool2d(x, 16, stride=8)
        detection4 = self.detector4(fm4)

        return detection1, detection2, detection3, detection4


if __name__ == '__main__':
    net = MyNet()
    torch.save(net.state_dict(), 'model/model')
