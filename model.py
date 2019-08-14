import torch
import torch.nn as nn


def make_layer(din, dout, size):
    padding = (size // 2)
    return torch.nn.Sequential(
        nn.Conv2d(din, dout, size, padding=padding, stride=1),
        nn.ReLU(inplace=True)
    )


class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self._net = torch.nn.Sequential(
            make_layer(3, 32, 3),
            make_layer(32, 32, 3),
            nn.MaxPool2d(2),  # 112
            make_layer(32, 64, 3),
            make_layer(64, 64, 3),
            nn.MaxPool2d(2),  # 56
            make_layer(64, 128, 3),
            make_layer(128, 128, 3),
            nn.MaxPool2d(2),  # 28
            make_layer(128, 256, 3),
            make_layer(256, 256, 3),
            make_layer(256, 256, 3),
            nn.MaxPool2d(2),  # 14
            make_layer(256, 256, 3),
            make_layer(256, 256, 3),
            make_layer(256, 256, 3),
            nn.MaxPool2d(2),  # 7
            make_layer(256, 256, 3),
            make_layer(256, 256, 3),
            make_layer(256, 256, 3),
            nn.Conv2d(256, 11, 1, padding=0, stride=1)
        )

    def forward(self, x):
        return self._net(x)


if __name__ == '__main__':
    model = MyNetwork()
    torch.save(model.state_dict(), 'model/model')

