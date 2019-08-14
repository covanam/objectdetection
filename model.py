import torch
import torch.nn as nn
import torch.nn.functional as func

def make_layer(din, dout, size):
    padding = (size // 2)
    return torch.nn.Sequential(
        nn.Conv2d(din, dout, size, padding=padding, stride=1),
        nn.ReLU(inplace=True)
    )


class MSNet(nn.Module):
    def _Detector(in_depth):
        return nn.Sequential(
            nn.Conv2d(base_feature_depth, 1024, 1, padding=0, stride=1, bias=True),
            nn.ReLU(True),
            nn.Conv2d(1024, 10, 1, padding=0, stride=1, bias=True)
        )

    def __init__(self, base_network, base_feature_depth, detector=MSNet._Detector):
        self.feature_extractor = base_network()
        self.detector1 = detector()

    def forward(self, x):
        features = self.feature_extractor(x)  # 16x16
        
        fm1 = func.avg_pool2d(x, 2, stride=1)
        detection1 = self.detector2(fm1)  # 15x15
        
        fm2 = func.avg_pool2d(x, 4, stride=2)
        detection2 = self.detector1(fm2)  # 7x7
        
        fm3 = func.avg_pool2d(x, 8, stride=4)
        detection3 = self.detector1(fm3)  # 3x3
        
        fm4 = func.avg_pool2d(x, 16, stride=8)
        detection4 = self.detector851(fm4)
        
        return (detection1, detection2, detection3, detection4)


if __name__ == '__main__':
    model = MyNetwork()
    torch.save(model.state_dict(), 'model/model')

