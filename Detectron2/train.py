"""
## Dependencies:
* torch
* torchvision
* pyyaml
* cython
* cocoapi
* detectron2
* opencv-python

## Getting dataset ready:
Extract the coco dataset into the content
"""

import torch
from detectron2.utils.logger import setup_logger
print(torch.__version__, torch.cuda.is_available())
setup_logger()

# import some common detectron2 utilities
import os
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances

"""# Import the dataset"""
register_coco_instances("my_dataset_train", {}, "content/train/_annotations.coco.json", "content/train")
register_coco_instances("my_dataset_val", {}, "content/valid/_annotations.coco.json", "content/valid")
register_coco_instances("my_dataset_test", {}, "content/test/_annotations.coco.json", "content/test")

"""# Configure Detectron"""

#We are importing our own Trainer Module here to use the COCO validation evaluation during training. Otherwise no validation eval occurs.

from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator

class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)

from detectron2.config import get_cfg

cfg = get_cfg() # Load the detectron config
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")) # Merge a default COCO config form model zoo
cfg.DATASETS.TRAIN = ("my_dataset_train",)  # Set which collection to use for training
cfg.DATASETS.TEST = ("my_dataset_val",) # Set which collection to use for testing

cfg.DATALOADER.NUM_WORKERS = 2 # How many threads to use for loading the data from the dataset

cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo

cfg.SOLVER.IMS_PER_BATCH = 2 # how many images to train per step, lower number means less memory used
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 500
cfg.SOLVER.MAX_ITER = 1000 #adjust up if val mAP is still rising, adjust down if overfit
cfg.SOLVER.STEPS = []
cfg.SOLVER.GAMMA = 0.05

cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 64 # Number of regions per image used to train RPN
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2 # The amount of classes in the dataset

cfg.TEST.EVAL_PERIOD = 500 # Will evaluate the model every X steps

if not torch.cuda.is_available():
    cfg.MODEL.DEVICE = "cpu"

"""# Training"""
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True) # Make sure the output folder nxists

trainer = CocoTrainer(cfg) # Register a CocoTrainer with our config
trainer.resume_or_load(resume=False) # If true, continues training from last checkpoint in output directory
trainer.train() # Start the training process

"""# Test the model"""
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)
evaluator = COCOEvaluator("my_dataset_test", cfg, False, output_dir="./output/")
val_loader = build_detection_test_loader(cfg, "my_dataset_test")
inference_on_dataset(trainer.model, val_loader, evaluator)


# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
# cfg.DATASETS.TEST = ("my_dataset_test", )
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
# predictor = DefaultPredictor(cfg)
# test_metadata = MetadataCatalog.get("my_dataset_test")