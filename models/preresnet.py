import torch.nn as nn
import torch.nn.functional as F
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False))

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self,
                 block,
                 num_blocks,
                 initial_channels,
                 num_classes,
                 stride=1,
                 width_multiplier=1.0,
                 depth_multiplier=1.0):
        super(PreActResNet, self).__init__()
        self.in_planes = initial_channels
        self.num_classes = num_classes
        #import pdb; pdb.set_trace()
        self.conv1 = nn.Conv2d(3,
                               initial_channels,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block,
                                       round_filters(initial_channels, width_multiplier),
                                       int(depth_multiplier * num_blocks[0]),
                                       stride=1)
        self.layer2 = self._make_layer(block,
                                       round_filters(initial_channels * 2, width_multiplier),
                                       int(num_blocks[1] * depth_multiplier),
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       round_filters(initial_channels * 4, width_multiplier),
                                       int(num_blocks[2] * depth_multiplier),
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       round_filters(initial_channels * 8, width_multiplier),
                                       int(num_blocks[3] * depth_multiplier),
                                       stride=2)
        self.linear = nn.Linear(
            round_filters(initial_channels * 8, width_multiplier) * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def compute_h1(self, x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        return out

    def compute_h2(self, x):
        out = x
        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

    def forward(self, x, profile=None):
        out = x

        out = self.conv1(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view(out.shape[0], -1)
        out = self.linear(out)

        return out


def preactresnet18(num_classes=10,
                   dropout=False,
                   stride=1,
                   width_multiplier=1.0,
                   depth_multiplier=1.0):
    return PreActResNet(PreActBlock, [2, 2, 2, 2],
                        64,
                        num_classes,
                        stride=stride,
                        width_multiplier=width_multiplier,
                        depth_multiplier=depth_multiplier)


def preactresnet34(num_classes=10,
                   dropout=False,
                   stride=1,
                   width_multiplier=1.0,
                   depth_multiplier=1.0):
    return PreActResNet(PreActBlock, [3, 4, 6, 3],
                        64,
                        num_classes,
                        stride=stride,
                        width_multiplier=width_multiplier,
                        depth_multiplier=depth_multiplier)


def preactresnet50(num_classes=10,
                   dropout=False,
                   stride=1,
                   width_multiplier=1.0,
                   depth_multiplier=1.0):
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3],
                        64,
                        num_classes,
                        stride=stride,
                        width_multiplier=width_multiplier,
                        depth_multiplier=depth_multiplier)


def preactresnet101(num_classes=10,
                    dropout=False,
                    stride=1,
                    width_multiplier=1.0,
                    depth_multiplier=1.0):
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3],
                        64,
                        num_classes,
                        stride=stride,
                        width_multiplier=width_multiplier,
                        depth_multiplier=depth_multiplier)


def preactresnet152(num_classes=10,
                    dropout=False,
                    stride=1,
                    width_multiplier=1.0,
                    depth_multiplier=1.0):
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3],
                        64,
                        num_classes,
                        stride=stride,
                        width_multiplier=width_multiplier,
                        depth_multiplier=depth_multiplier)


def round_filters(filters, multiplier=None):
    """Calculate and round number of filters based on width multiplier.
    Args:
        filters (int): Filters number to be calculated.
        multiplier (float): multiplier for width scale.
    Returns:
        new_filters: New filters number after calculating.
    """

    if multiplier is None:
        return filters
    # TODO: modify the params names.
    #       maybe the names (width_divisor,min_width)
    #       are more suitable than (depth_divisor,min_depth).

    divisor = 8
    min_depth = None
    filters *= multiplier
    min_depth = min_depth or divisor  # pay attention to this line when using min_depth
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)
