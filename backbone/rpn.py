from typing import Dict, List
import torch
import torch.nn.functional as F
import torch.nn as nn

from detectron2.layers import ShapeSpec
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY, RPN_HEAD_REGISTRY
from detectron2.structures import ImageList


@RPN_HEAD_REGISTRY.register()
class SeparableRPNHead(nn.Module):

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(
            set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_cell_anchors = anchor_generator.num_cell_anchors
        box_dim = anchor_generator.box_dim
        assert (len(set(num_cell_anchors)) == 1
               ), "Each level must have the same number of cell anchors"
        num_cell_anchors = num_cell_anchors[0]

        # large separable conv for the hidden representation
        # NOTE: this procedure should be done on the feature maps produced by FPN.
        # 1. large kernel size for better loc?
        # 2. separable conv to save parameters
        mid_channels = cfg.MODEL.RPN.MID_CHANNELS
        out_channels = in_channels  # TODO
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=(7, 1),
                      stride=1,
                      padding=(3, 0)),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=(1, 7),
                      stride=1,
                      padding=(0, 3)),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=(1, 7),
                      stride=1,
                      padding=(0, 3)),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=(7, 1),
                      stride=1,
                      padding=(3, 0)),
            nn.ReLU(inplace=True),
        )

        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(in_channels,
                                           num_cell_anchors,
                                           kernel_size=1,
                                           stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(in_channels,
                                       num_cell_anchors * box_dim,
                                       kernel_size=1,
                                       stride=1)
        for branch in [self.branch1, self.branch2]:
            for m in branch.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.constant_(m.bias, 0)

        for l in [self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = torch.add(self.branch1(x), self.branch2(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas
