import torch
import torch.nn as nn

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
from detectron2.modeling.backbone.fpn import LastLevelMaxPool

from .fpn import DepthwiseFPN


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):

    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp,
                                    inp,
                                    kernel_size=5,
                                    stride=self.stride,
                                    padding=2),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp,
                          branch_features,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features,
                                branch_features,
                                kernel_size=5,
                                stride=self.stride,
                                padding=2),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features,
                      branch_features,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i,
                         o,
                         kernel_size,
                         stride,
                         padding,
                         bias=bias,
                         groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(Backbone):

    def __init__(self,
                 stages_repeats,
                 stages_out_channels,
                 num_classes=None,
                 out_features=None,
                 inverted_residual=InvertedResidual):
        super(ShuffleNetV2, self).__init__()
        self.num_classes = num_classes

        current_stride = 4  # = stride 2 conv -> stride 2 max pool
        self._out_feature_strides = {"stem": current_stride}
        self._out_feature_channels = {"stem": stages_out_channels[0]}

        if len(stages_repeats) != 4:
            raise ValueError(
                'expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 6:
            raise ValueError(
                'expected stages_out_channels as list of 5 positive ints')

        input_channels = 3
        output_channels = stages_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stages_and_names = []
        for i, (repeats, output_channels) in enumerate(
                zip(stages_repeats, stages_out_channels[1:])):
            # NOTE: the stride in the first stage
            first_stride = 1 if i == 0 else 2
            seq = [
                inverted_residual(input_channels, output_channels, first_stride)
            ]
            for _ in range(repeats - 1):
                seq.append(
                    inverted_residual(output_channels, output_channels, 1))
            stage = nn.Sequential(*seq)
            name = "stage" + str(i + 2)
            self.add_module(name, stage)
            self.stages_and_names.append((stage, name))
            # TODO hard code here!
            self._out_feature_strides[
                name] = current_stride = current_stride * first_stride
            self._out_feature_channels[name] = output_channels
            input_channels = output_channels

        if self.num_classes is not None:
            output_channels = stages_out_channels[-1]
            self.conv5 = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

            self.fc = nn.Linear(output_channels, num_classes)

            nn.init.normal_(self.fc.weight, std=0.01)
            name = "linear"

        if out_features is None:
            out_features = [name]
        self._out_features = out_features
        assert len(self._out_features)
        children = [x[0] for x in self.named_children()]
        for out_feature in self._out_features:
            assert out_feature in children, "Available children: {}".format(
                ", ".join(children))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        outputs = {}
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.maxpool(x)
        if "stem" in self._out_features:
            outputs["stem"] = x
        for stage, name in self.stages_and_names:
            x = stage(x)
            if name in self._out_features:
                outputs[name] = x
        if self.num_classes is not None:
            x = self.conv5(x)
            x = x.mean([2, 3])  # globalpool
            x = self.fc(x)
            if "linear" in self._out_features:
                outputs["linear"] = x
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            ) for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_shufflenetv2_backbone(cfg, input_shape=None):
    """
    Create a ShuffleNetV2 instance from config.

    Returns:
        ResNet: a :class:`ShuffleNetV2` instance.
    """

    # TODO
    configs = {
        "x0.5": [[2, 4, 8, 4], [24, 24, 48, 96, 192, 1024]],
        "x1.0": [[2, 4, 8, 4], [24, 24, 116, 232, 464, 1024]],
        "x1.5": [[2, 4, 8, 4], [24, 24, 176, 352, 704, 1024]],
        "x2.0": [[2, 4, 8, 4], [24, 24, 244, 488, 976, 2048]],
    }

    out_features = cfg.MODEL.SHUFFLENETS.OUT_FEATURES
    depth_multiplier = cfg.MODEL.SHUFFLENETS.DM

    # TODO
    repeats, out_channels = configs[depth_multiplier]

    return ShuffleNetV2(repeats, out_channels, out_features=out_features)


@BACKBONE_REGISTRY.register()
def build_shufflenetv2_fpn_backbone(cfg, input_shape=None):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_shufflenetv2_backbone(cfg)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = DepthwiseFPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )

    return backbone


if __name__ == "__main__":
    from .config import cfg
    model = build_shufflenetv2_fpn_backbone(cfg)
    print(model)

    x = torch.randn(2, 3, 480, 800)
    y = model(x)
    print(y.keys())
    print([x.shape for x in y.values()])