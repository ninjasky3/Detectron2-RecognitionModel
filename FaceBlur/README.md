# FaceBlurring

### Requirements
* Linux or macOS with Python ≥ 3.7 with an environment that supports pip
* PyTorch ≥ 1.8 and torchvision that matches the PyTorch installation. Install them together at pytorch.org to make sure of this
* Latest version of Detectron2 (see below for additional info)
* OpenCV via the `opencv-python` package
* Trained facedetection Detectron2 model(see train.py)

#### Installing Detectron
gcc & g++ ≥ 5.4 are required. ninja is optional but recommended for faster build. After having them, run:

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2

# On macOS, you may need to prepend the above commands with a few environment variables:
CC=clang CXX=clang++ ARCHFLAGS="-arch x86_64" python -m pip install ...
```

To rebuild detectron2 that’s built from a local clone, use rm -rf build/ **/*.so to clean the old build first. You often need to rebuild detectron2 after reinstalling PyTorch.

## Usage
To use the FaceDetection you also need to train or download the model, and place it somewhere on your PC.

After this is done you need to import faceBlurring from FaceDetection set the model path and instantiate faceBlurring with the path and accuracy treshhold: 

```from FaceDetection import faceBlurring

directory = os.path.normpath(os.getcwd() + os.sep + os.pardir)
path = directory + "/Detectron2/output/model_final.pth" #edit this path with where your model is saved

# Create instance of faceBlurring
detector = faceBlurring(path,accuracy)  

# sys.argv[1] is the first argument you give when running FaceBlurringExample.py
# example python3 FaceBlurringExample.py "/Detectron2/output/model_final.pth"

detector.runDetection(sys.argv[1])
```
