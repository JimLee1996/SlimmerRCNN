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
cfg.MODEL.SHUFFLENETS.DM = "x1.0"
cfg.MODEL.SHUFFLENETS.OUT_FEATURES = ["stage2", "stage3", "stage4"]
cfg.MODEL.FPN.IN_FEATURES = ["stage2", "stage3", "stage4"]
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[64], [128], [256], [512]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
cfg.MODEL.RPN.IN_FEATURES = ["p3", "p4", "p5", "p6"]
cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p3", "p4", "p5"]
