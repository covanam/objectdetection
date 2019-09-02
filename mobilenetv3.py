import torch.nn as nn
import torch.nn.functional as F
import torch
import time

def conv_bn(inp, oup, stride, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 3, stride, 1, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class UnNamed(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.abs() * x.tanh()


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
            # nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()

        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            se_layer = SEModule
        else:
            se_layer = Identity

        self.conv = nn.Sequential(
            # point wise conv
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # depth wise conv
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            se_layer(exp),
            nlin_layer(inplace=True),
            # point wise, no non-linearity
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp):
        super().__init__()

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(
            # point wise conv
            conv_layer(inp, exp, 1, 1, 0, bias=True),
            norm_layer(exp),
            UnNamed(),
            # depth wise conv
            conv_layer(exp, exp, kernel, stride, 0, groups=exp, bias=False),
            norm_layer(exp),
            UnNamed(),
            conv_layer(exp, oup, 1, 1, 0, bias=True)
        )

    def forward(self, x):
        return self.conv(x)


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        mobile_setting = [
            # k, exp, c,  se,     nl,  s,
            [3, 16,  16,  False, 'RE', 1],
            [2, 64,  24,  False, 'RE', 2],  # kernel 3 -> 2
            [3, 72,  24,  False, 'RE', 1],
            [4, 72,  40,  True,  'RE', 2],  # kernel 5 -> 4
            [5, 120, 40,  True,  'RE', 1],
            [5, 120, 40,  True,  'RE', 1],
            [2, 240, 80,  False, 'HS', 2],  # kernel 3 -> 2
            [3, 200, 80,  False, 'HS', 1],
            [3, 184, 80,  False, 'HS', 1],
            [3, 184, 80,  False, 'HS', 1],
            [3, 480, 112, True,  'HS', 1],
            [3, 672, 112, True,  'HS', 1],
            # [5, 672, 160, True,  'HS', 1],  # removed
            # [5, 960, 160, True,  'HS', 1],  # removed
            # [5, 960, 160, True,  'HS', 1],  # removed
        ]

        # building first layer
        self.features = [conv_bn(3, 16, 2, nlin_layer=Hswish)]

        # building mobile blocks inp, oup, kernel, stride, exp, se=False, nl='RE'):
        cin = 16
        for k, exp, c, se, nl, s in mobile_setting:
            self.features.append(MobileBottleneck(cin, c, k, s, exp, se, nl))
            cin = c

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        self.decoder1 = Decoder(112, 50, 2, 1, 1024)
        self.decoder2 = Decoder(112, 50, 4, 2, 512)
        self.decoder3 = Decoder(112, 50, 8, 4, 256)
        self.decoder4 = Decoder(112, 50, 16, 8, 128)

        self._initialize_weights()

    def forward(self, x):
        tick = time.time()
        feature_map = self.features(x)
        print(time.time() - tick)

        tick = time.time()
        detection1 = self.decoder1(feature_map)
        print(time.time() - tick)

        tick = time.time()
        detection2 = self.decoder2(feature_map)
        print(time.time() - tick)

        tick = time.time()
        detection3 = self.decoder3(feature_map)
        print(time.time() - tick)

        tick = time.time()
        detection4 = self.decoder4(feature_map)
        print(time.time() - tick)

        return detection1, detection2, detection3, detection4

    def _initialize_weights(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


if __name__ == '__main__':
    net = MyNet().eval()
    torch.save(net.state_dict(), 'model/model')
    x = torch.zeros((1, 3, 256, 256))
    torch.set_num_threads(1)
    with torch.no_grad():
        start = time.time()
        pred = net(x)
        print(time.time() - start)