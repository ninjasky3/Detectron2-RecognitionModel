from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import torch
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import numpy as np

import cv2

import sys
import os

class faceBlurring:

    def __init__(self, model_path="Detectron2/output/model.pth",
                 accuracy_threshold=0.96,
                 num_classes=2
                 ):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = accuracy_threshold  # Set threshold for this model
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes

        if not torch.cuda.is_available():
            cfg.MODEL.DEVICE = "cpu"

        self.predictor = DefaultPredictor(cfg)


    """
    This code defines a function called "get_images" which takes in one parameter:

    "path": a string representing the path to a directory containing the images to be processed.
    The function starts by initializing an empty list called "images". 
    It then iterates through all the files in the directory specified by the "path" parameter. 
    It then check if the file ends with '.png' and if it does, it will append the file to the list of images.

    Finally, the function returns the list of images.

    The code also includes a try-except block that attempts to get the second command line argument as a float and assigns it to the variable "threshold", 
    with a default value of 0.80 if it fails.
    """
    def get_images(self,path: str):
        images = []
        for file in os.listdir(path):
            if (file.endswith(".png")):
                images.append(file)
        
        return images

    try:
        threshold = float(sys.argv[2])
    except:
        threshold = 0.80

    """
    This code defines a function called "blurDetectedFaces" which takes in two parameters:

    "image": The input image on which the detection was performed.
    "box": A list of four elements representing the coordinates of the bounding box in the image. [x_top_left, y_top_left, x_bottom_right, y_bottom_right]
    The function starts by extracting the coordinates of the bounding box from the input list. Then it cast them to integer and crops the image according to the bounding box coordinates.

    After that, it applies a Gaussian blur on the cropped image with a kernel of (23, 23) and standard deviation of 30.

    Finally, it replaces the cropped part of the original image with the blurred one and returns the output image.
    """

    def blurDetectedFaces(self,image, box):
        x_top_left = box[0]
        y_top_left = box[1]
        x_bottom_right = box[2]
        y_bottom_right = box[3]


        xtlint = int(float(x_top_left))
        ytlint = int(float(y_top_left))
        xbrint = int(float(x_bottom_right))
        ybrint = int(float(y_bottom_right))

        crop_img= image[ytlint:ybrint, xtlint:xbrint]

        blurred_img = cv2.GaussianBlur(crop_img, (23, 23), 30)
        image[ytlint:ybrint, xtlint:xbrint] = blurred_img
        return image

    """
    This code runs object detection on a set of images, 
    detects the bounding boxes for the objects and then applies a blur effect to those regions in the images, 
    then it shows the images in which the detected objects are blurred.

    The function runDetection(self, imagePath) takes in one parameter, "imagePath", which is a string representing the path to a directory containing the images to be processed.
    test_metadata = MetadataCatalog.get("my_dataset_test"): the metadata for the dataset being used for the object detection is obtained.
    """

    def runDetection(self,imagePath):
        test_metadata = MetadataCatalog.get("my_dataset_test")
        img_nr = 0
        images = self.get_images(imagePath)
        for img in images:
            print(str(img_nr)+"/"+str(len(images))) 
            im = cv2.imread(imagePath+"/"+img)
            outputs = self.predictor(im)
            v = Visualizer(im[:, :, ::-1],
                            metadata=test_metadata, 
                            scale=0.8
                            )
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            boxes = outputs["instances"].pred_boxes.tensor.tolist()        
            box = list(boxes)
            for index, item in enumerate(boxes):
                box[index] = boxes[index]
        for i,item in enumerate(box):
            BlurredResult = self.blurDetectedFaces(im, item)
            cv2.imshow("", BlurredResult)
            cv2.waitKey(1000)



