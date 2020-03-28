import torch
from detectron2.modeling import build_backbone, build_proposal_generator, build_roi_heads

from backbone import cfg
from backbone import build_shufflenetv2_fpn_backbone

backbone_model = build_backbone(cfg)
print(backbone_model)
torch.save(backbone_model.state_dict(), '1.backbone.pth')

proposal_model = build_proposal_generator(cfg, backbone_model.output_shape())
print(proposal_model)
torch.save(proposal_model.state_dict(), '2.rpn.pth')

roi_model = build_roi_heads(cfg, backbone_model.output_shape())
print(roi_model)
torch.save(roi_model.state_dict(), '3.roi.pth')
