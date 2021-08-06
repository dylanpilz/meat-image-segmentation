# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 11:49:21 2021

@author: PilzD

adapted from:
https://github.com/keras-team/keras-io/blob/master/examples/vision/oxford_pets_image_segmentation.py

"""

import os
import matplotlib.pyplot as plt


input_dir = "//hvcimlab/CSBVISION01/CSB-TransportUnitRecognition/MTU/TF/Oxford_pets/images/"
target_dir = "//hvcimlab/CSBVISION01/CSB-TransportUnitRecognition/MTU/TF/Oxford_pets/annotations/trimaps/"
img_size = (160, 160)
num_classes = 3
batch_size = 32

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith(".jpg")
    ]
)
target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, "|", target_path)

#%%

import cv2, os
def jpeg_converter(srcdir, newdir):
    import cv2, os
    for infile in os.listdir(srcdir):
        print ("file : " + infile)
        read = cv2.imread(srcdir + infile)
        outfile = infile.split('.')[0] + '.jpg'
        cv2.imwrite(newdir+outfile,read,[int(cv2.IMWRITE_JPEG_QUALITY), 200])
#%%

from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps

# Display input image #7
display(Image(filename=input_img_paths[9]))

# Display auto-contrast version of corresponding target (per-pixel categories)
img = PIL.ImageOps.autocontrast(load_img(target_img_paths[9]))
display(img)


#%%

from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import load_img


class DataGen(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i : i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode="grayscale")
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
            y[j] -= 1
        return x, y


#%%


from tensorflow.keras import layers


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()

# Build model
model = get_model(img_size, num_classes)
model.summary()

#%%

import random

# Split our img paths into a training and a validation set
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[:-val_samples]
train_target_img_paths = target_img_paths[:-val_samples]
val_input_img_paths = input_img_paths[-val_samples:]
val_target_img_paths = target_img_paths[-val_samples:]

# Instantiate data Sequences for each split
train_gen = DataGen(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = DataGen(batch_size, img_size, val_input_img_paths, val_target_img_paths)


#%%
import tensorflow as tf
import datetime

model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
callbacks = [
    keras.callbacks.ModelCheckpoint("oxford_segmentation.h5", save_best_only=True),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]

# Train the model, doing validation at the end of each epoch.
epochs = 500
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)

#%%
model.save("//srvpnw/Volp5/CSB-Vision/temp/4DPI/Praktikum/Oxford_pets/Model (pets)/Model.h5")

print(type(model))
#%%
import tensorflow as tf   
print(tf.__version__)
from keras.models import load_model

model_loaded = load_model("//srvpnw/Volp5/CSB-Vision/temp/4DPI/Praktikum/Oxford_pets/Model (pets)/model.h5")

model_loaded.summary()
print(type(model_loaded))


#%%
# Generate predictions for all images in the validation set
import cv2 as cv
import os
path = "//srvpnw/Volp5/CSB-Vision/temp/4DPI/Praktikum/Segmentation_examples/crates_jpg/"


animal_img_paths = sorted(
    [
        os.path.join(path, fname)
        for fname in os.listdir(path)
        if fname.endswith(".jpg")
    ]
)

val_gen = DataGen(batch_size, img_size,
                     animal_img_paths,
                     val_target_img_paths)

val_preds = model_loaded.predict(img for img in val_gen)

def display_mask(i, path):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    p = "//srvpnw/Volp5/CSB-Vision/temp/4DPI/Praktikum/Segmentation_examples/muscles_seg/"
    #cv.imwrite(p + path.split("_jpg/")[1], np.float32(img))
    display(img)

# Display results for validation image
for i in range(len(animal_img_paths)):
    # Display input image
    display(Image(filename=animal_img_paths[i]))
    '''
    # Display ground-truth target mask
    img = PIL.ImageOps.autocontrast(load_img(animal_img_paths[i]))
    display(img)
    '''
    # Display mask predicted by our model
    display_mask(i, animal_img_paths[i])  # Note that the model only sees inputs at 150x150.
