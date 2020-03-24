from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN

# 生成默认配置
cfg = get_cfg()

# 设置基本模型
base_model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"

# 使用fater-rcnn RPN配置
cfg.merge_from_file(model_zoo.get_config_file(base_model))

# SHUFFLENET BASE CONFIG
cfg.MODEL.BACKBONE.NAME = "build_shufflenetv2_fpn_backbone"

cfg.MODEL.SHUFFLENETS = CN()
cfg.MODEL.SHUFFLENETS.DM = "large"
cfg.MODEL.SHUFFLENETS.OUT_FEATURES = ["stage2", "stage3", "stage4", "stage5"]

cfg.MODEL.FPN.IN_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
cfg.MODEL.FPN.OUT_CHANNELS = 128

cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

cfg.MODEL.RPN.HEAD_NAME = "SeparableRPNHead"
cfg.MODEL.RPN.IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
cfg.MODEL.RPN.MID_CHANNELS = 64  # cfg.MODEL.FPN.OUT_CHANNELS / 2

cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"

cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 5
cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 1
cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 256
cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 1
cfg.MODEL.ROI_BOX_HEAD.CONV_DIM = 64
