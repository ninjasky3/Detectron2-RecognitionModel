# Detectron2 train and run scripts

This folder contains two scripts to train (train.py) and run (run.py) detectron2 on a COCO dataset.

## Requirements
* Linux or macOS with Python ≥ 3.7 with an environment that supports pip
* PyTorch ≥ 1.8 and torchvision that matches the PyTorch installation. Install them together at pytorch.org to make sure of this
* Latest version of Detectron2 (see below for additional info)
* OpenCV via the `opencv-python` package
* tqdm

### Installing Detectron
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

## How to setup and train

Extract any given COCO dataset into the subfolder called `content`. It should have the following structure:
```
content/valid/[images.jpg | _annotations.coco.json]
content/train/[images.jpg | _annotations.coco.json]
content/test/[images.jpg | _annotations.coco.json]
```

Afterwards make sure that in train.py and run.py the relevant settings like number of classes represent the values in your chosen dataset.

After this is done you can train the model by running `python train.py` in a terminal.

If your GPU does not have enough memory, you can adjust the `cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE` value.

## How to run the trained model on an image on your pc

After you have a trained model, you can run it using the following command `python run.py <Image_to_run_on>`, and example of a valid command: `python run.py test.jpg`. This script will then load the model and predict on the image. The result will be saved in the same folder as the original image with the same filename, but with `-out.png` appended to it. So the result of `test.jpg` would become `test.jpg-out.png`
