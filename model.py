import torch
import torch.nn as nn
import torch.nn.functional as func
import mnasnet


class MyNet(nn.Module):
    @staticmethod
    def _detector():
        return nn.Sequential(
            nn.Conv2d(1280, 2409, 1, padding=0, stride=1, bias=True),
            nn.Dropout2d(inplace=True, p=0),
            nn.ReLU(True),
            nn.Conv2d(2409, 2409, 1, padding=0, stride=1, bias=True),
            nn.Dropout2d(inplace=True, p=0),
            nn.ReLU(True),
            nn.Conv2d(2409, 14, 1, padding=0, stride=1, bias=True)
        )

    def __init__(self, base_network=mnasnet.MNASNet(1.0)):
        super().__init__()
        self.feature_extractor = base_network
       
        self.detector1 = self._detector()
        self.detector2 = self._detector()
        self.detector3 = self._detector()
        self.detector4 = self._detector()

    def forward(self, x):
        feature = self.feature_extractor  # 16x16
        x = feature(feature, x)
        
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
        return x

def model():
    base_net = mnasnet.MNASNet(1.0)
    del base_net.classifier
    base_net.forward = _forward
    
    model = MyNet(base_net)
    model.load_state_dict(torch.load('model/model'))

    return model

if __name__ == '__main__':
    base_net = mnasnet.MNASNet(1.0)
    base_net.load_state_dict(torch.load('model/pretrained'))
    del base_net.classifier
    base_net.forward = _forward
    net = MyNet(base_net)
    torch.save(net.state_dict(), 'model/model')
