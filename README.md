# meat-image-segmentation

# Overview
In this project, I used a (slightly modified) U-net model, trained on the Oxford-IIT Pets dataset in order to perform image segmentation on muscle images taken from CSB's JUMBOFLASH hardware.

# Files
## meat-segmentation.py
Subsets data, prepares and trains model. 
Model adapted from https://github.com/keras-team/keras-io/blob/master/examples/vision/oxford_pets_image_segmentation.py

## Model.h5
Trained net, using Oxford PETS dataset over 500 training epochs
