import torch
import torch.nn as nn

# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997


class _InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, stride, expansion_factor,
                 bn_momentum=0.1):
        super().__init__()
        mid_ch = in_ch * expansion_factor
        self.apply_residual = (in_ch == out_ch and stride == 1)
        self.layers = nn.Sequential(
            # Pointwise
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_ch, mid_ch, kernel_size, padding=(kernel_size - 1) // 2,
                      stride=stride, groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # Linear pointwise. Note that there's no activation.
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, momentum=bn_momentum))

    def forward(self, input):
        if self.apply_residual:
            return self.layers(input) + input
        else:
            return self.layers(input)


def _stack(in_ch, out_ch, kernel_size, stride, exp_factor, repeats,
           bn_momentum):
    """ Creates a stack of inverted residuals. """
    assert repeats >= 1
    # First one has no skip, because feature map size changes.
    first = _InvertedResidual(in_ch, out_ch, kernel_size, stride, exp_factor,
                              bn_momentum=bn_momentum)
    remaining = []
    for _ in range(1, repeats):
        remaining.append(
            _InvertedResidual(out_ch, out_ch, kernel_size, 1, exp_factor,
                              bn_momentum=bn_momentum))
    return nn.Sequential(first, *remaining)


class _Eye(torch.nn.Module):
    class LastLayer(nn.Module):
        def __init__(self, in_ch, out_ch, expansion_factor,
                     bn_momentum=0.1):
            super().__init__()
            mid_ch = in_ch * expansion_factor
            self.layers = nn.Sequential(
                # Pointwise
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
                nn.ReLU(inplace=True),
                # Depthwise
                nn.Conv2d(mid_ch, mid_ch, 4, padding=0, stride=2, groups=mid_ch, bias=False),
                nn.BatchNorm2d(mid_ch, momentum=bn_momentum),
                nn.ReLU(inplace=True),
                # Linear pointwise
                nn.Conv2d(mid_ch, out_ch, 1, bias=True)  # note on bias
            )

        def forward(self, x):
            return self.layers(x)

    def __init__(self, in_ch=96, out_ch=50, bn_momentum=_BN_MOMENTUM):
        super().__init__()
        self.layers = nn.Sequential(
            _InvertedResidual(in_ch, in_ch, 3, 1, 6, bn_momentum),
            _InvertedResidual(in_ch, in_ch, 3, 1, 6, bn_momentum),
            _Eye.LastLayer(in_ch, out_ch, 6, bn_momentum)
        )

    def forward(self, x):
        return self.layers(x)


class _Rescale(nn.Module):
    def __init__(self, c, bn_momentum=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            _InvertedResidual(c, c, 2, 2, 6, bn_momentum),
            _InvertedResidual(c, c, 3, 1, 6, bn_momentum),
            _InvertedResidual(c, c, 3, 1, 6, bn_momentum)
        )

    def forward(self, x):
        return self.layers(x)


class MNASNet(torch.nn.Module):
    """ MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf.
    >>> model = MNASNet(1000, 1.0)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    1
    >>> y.nelement()
    1000
    """

    def __init__(self):
        super(MNASNet, self).__init__()
        depths = [24, 40, 80, 96, 192, 320]
        layers = [
            # First layer: regular conv.
            nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # Depthwise separable, no skip.
            nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32, bias=False),
            nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(16, momentum=_BN_MOMENTUM),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(16, depths[0], 3, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[0], depths[1], 5, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[1], depths[2], 5, 2, 6, 3, _BN_MOMENTUM),
            _stack(depths[2], depths[3], 3, 1, 6, 2, _BN_MOMENTUM)
        ]

        self.layers = nn.Sequential(*layers)

        self.rescale1 = _Rescale(96, _BN_MOMENTUM)
        self.rescale2 = _Rescale(96, _BN_MOMENTUM)
        self.rescale3 = _Rescale(96, _BN_MOMENTUM)

        self.eye1 = _Eye()
        self.eye2 = _Eye()
        self.eye3 = _Eye()
        self.eye4 = _Eye()

    def forward(self, x):
        with torch.no_grad():
            ft1 = self.layers(x)
        ft2 = self.rescale1(ft1)
        ft3 = self.rescale2(ft2)
        ft4 = self.rescale3(ft3)

        d_1 = self.eye1(ft1)
        d_2 = self.eye2(ft2)
        d_3 = self.eye3(ft3)
        d_4 = self.eye4(ft4)

        return d_1, d_2, d_3, d_4

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)


if __name__ == '__main__':
    import time

    net = MNASNet().eval()
    pretrained = torch.load('model/mnasnet.pth')
    for key in net.layers.state_dict().keys():
        key = 'layers.' + key
        net.state_dict()[key] = pretrained[key]

    x = torch.zeros((1, 3, 256, 384))

    st = time.time()
    with torch.no_grad():
        pred = net(x)
    print(time.time() - st)
    for p in pred:
        print(p.shape)

