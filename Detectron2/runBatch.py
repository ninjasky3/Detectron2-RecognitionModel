from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

import cv2
import torch
import sys
import os

def get_images(path: str):
    images = []
    for file in os.listdir(path):
        if (file.endswith(".png")):
            images.append(file)
    
    return images

images = get_images(sys.argv[1])

try:
    threshold = float(sys.argv[2])
except:
    threshold = 0.95

# Run as follows: python runBatch.py some_folder accuracy

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold # Set threshold for this model
cfg.MODEL.WEIGHTS = 'output/model_final.pth' # Set path model .pth
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2

if not torch.cuda.is_available():
    cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)

test_metadata = MetadataCatalog.get("my_dataset_test")

img_nr = 0

for img in images:
    print(str(img_nr)+"/"+str(len(images))) 
    im = cv2.imread(sys.argv[1]+"/"+img)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                    metadata=test_metadata, 
                    scale=0.8
                    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow("", out.get_image()[:, :, ::-1])
    cv2.waitKey(1000)

    #cv2.imwrite(sys.argv[1]+"/out/"+str(img_nr)+".png", out.get_image()[:, :, ::-1])
    img_nr += 1


