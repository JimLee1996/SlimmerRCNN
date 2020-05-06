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

# FPN
cfg.MODEL.FPN.IN_FEATURES = ["stage2", "stage3", "stage4", "stage5"]
cfg.MODEL.FPN.OUT_CHANNELS = 128

# ROI
cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p2", "p3", "p4", "p5"]
cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"

# Box Regression
cfg.MODEL.ROI_BOX_HEAD.NAME = "FastRCNNConvFCHead"
cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 5
cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"
cfg.MODEL.ROI_BOX_HEAD.NUM_CONV = 1
cfg.MODEL.ROI_BOX_HEAD.CONV_DIM = 32
cfg.MODEL.ROI_BOX_HEAD.NUM_FC = 1
cfg.MODEL.ROI_BOX_HEAD.FC_DIM = 512

cfg.INPUT.MAX_SIZE_TEST = 608
cfg.INPUT.MIN_SIZE_TEST = 608
cfg.INPUT.MAX_SIZE_TRAIN = 608
cfg.INPUT.MIN_SIZE_TRAIN = (544, 576, 608, 640)
