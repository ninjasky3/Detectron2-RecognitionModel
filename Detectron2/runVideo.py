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

from tqdm import tqdm

import time

# Use: copy this in your terminal(remove or fill in <> signs): "python3 runVideo.py /path/to/your/video/video.mp4 <threshold value> <frame skip value>"
# Default values: threshold: 0.95, frameSkip: 5

def main():
    threshold = set_threshold()
    frameSkip = set_frame_skip() 

    imgArray, size = get_vid_frames(threshold, frameSkip)
    make_avi(imgArray, size)
    convert_avi_to_mp4()
    
#Set detection threshold, for detecting objects with higher or lower certainty
def set_threshold():
    try:
        threshold = float(sys.argv[2])
    except:
        threshold = 0.95
    return threshold
    
#Set frame skip, to increase video speed and decrease processing time
def set_frame_skip():
    try:
        frameSkip = float(sys.argv[3])

        if frameSkip == 0:
            frameSkip = 1
    except:
        frameSkip = 5
    return frameSkip

def get_vid_frames(threshold, frameSkip):

    imgArray = []

    #Get model config, obtained by running train.py
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml'))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    cfg.MODEL.WEIGHTS = 'output/model_final.pth' 
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    
    if not torch.cuda.is_available():
        cfg.MODEL.DEVICE = "cpu"

    predictor = DefaultPredictor(cfg)
    test_metadata = MetadataCatalog.get("my_dataset_test")

    #Get video and first frame
    vidcap = cv2.VideoCapture(sys.argv[1])
    success,image = vidcap.read()

    #Get video length and shape
    vidLength = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    height, width, layers = image.shape
    size = (width,height)

    #For loop which draws bounding boxes on a given frame, then adds it to an array
    for count in tqdm (range(vidLength), desc="Analysing frames...",ascii=False, ncols=75):
        if(count%frameSkip == 0):
            im = image
            outputs = predictor(im)
            v = Visualizer(im[:, :, ::-1],metadata=test_metadata,scale=1)
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            imgArray.append(out.get_image()[:, :, ::-1])

            #cv2.imwrite("/home/niek/Images/frame%d.jpg" % count, image)     # save frame as JPEG file
            #cv2.imshow("Image", out.get_image()[:, :, ::-1]) 

        success,image = vidcap.read()
    return imgArray, size

#Make .avi file by converting image array 
def make_avi(imgArray, size):
    vid = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in tqdm (range(len(imgArray)), desc="Creating .avi...",ascii=False, ncols=75):
        vid.write(imgArray[i])

    vid.release()

# Delete previous video and convert .avi to .mp4
def convert_avi_to_mp4():

    if os.path.exists("out.mp4"):
        os.remove("out.mp4")
    else:
        print("ERROR: out.mp4 does not exist") 
    
    time.sleep(5)
    os.popen("ffmpeg -i project.avi -c:v libx264 -crf 19 -preset slow -c:a libfdk_aac -b:a 192k -ac 2 out.mp4")
    time.sleep(5)

    if os.path.exists("project.avi"):
        os.remove("project.avi")
    else:
        print("ERROR: project.avi does not exist") 

if __name__=="__main__":
    main()







    
