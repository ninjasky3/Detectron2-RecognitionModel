# Load the supplied image
import os
import sys

import cv2

from FaceDetection import faceBlurring

"""
# Example of using the package

Instantiate an instance of FaceDetection with the relevant info

Test the various methods

```
python FaceBlurring.py
```
"""
#to get the current working directory
directory = os.path.normpath(os.getcwd() + os.sep + os.pardir)
path = directory + "/Detectron2/output/model_final.pth" 

print(path)

modelPath= path
accuracy = 0.80


detector = faceBlurring(path,accuracy)  # Create instance of MastDetector

detector.runDetection(sys.argv[1])
