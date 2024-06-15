from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import cv2
import torch
import sys

try:
    threshold = float(sys.argv[2])
except:
    threshold = 0.95

# Run as follows: python run.py some_img.png accuracy

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # Set threshold for this model
cfg.MODEL.WEIGHTS = 'output/model_final.pth' # Set path model .pth
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

if not torch.cuda.is_available():
    cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

test_metadata = MetadataCatalog.get("my_dataset_test")

im = cv2.imread(sys.argv[1])
outputs = predictor(im)
v = Visualizer(im[:, :, ::-1],
                metadata=test_metadata, 
                scale=0.8
                 )
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2.imwrite(sys.argv[1]+".out.png", out.get_image()[:, :, ::-1])
