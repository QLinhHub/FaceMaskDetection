#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:36:36 2020

@author: quanglinhle
"""
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.applications import MobileNetV2

from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, 
                help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="learning_curve.png",
                help="path to loss/accuracy curve")
ap.add_argument("-m", "--model", type=str, default="mask_detector.model",
                help="path to output face mask detector model")
args = vars(ap.parse_args())

#initialize the initial learning rate, number of epochs and batch size
INIT_LR = 1e-4
EPOCHS = 5
BATCH_SIZE = 8

#grab the list of images in our dataset directory, then initialize the list of data, and class images
print("[INFO] loading images...")
imagePaths = list(paths.list_images(args["dataset"]))

data = []
labels = []

for imagePath in imagePaths:
    #extract the class labels from the filename
    label = imagePath.split(os.path.sep)[-2]
    
    #load the input image, size 224x224 and preprocess it
    image = load_img(imagePath, target_size=(224,224))
    image = img_to_array(image)
    image = preprocess_input(image)
    
    data.append(image)
    labels.append(label)
    
data = np.array(data, dtype="float32")
labels = np.array(labels)

#perform one-hot encoding on the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#split data to 80% for training and 20% for testing
(X_train, X_test, y_train, y_test) = train_test_split(data, labels, 
                                                      test_size=0.2, stratify=labels)

#construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#load the MobileNetV2, left off some head layers
base_model = MobileNetV2(weights="imagenet", include_top=False, 
                         input_tensor=tf.keras.Input(shape=(224,224,3)))

#construct the head layer will be placed on top of the base model
head_model = base_model.output
head_model = tf.keras.layers.AveragePooling2D(pool_size=(7,7))(head_model)
head_model = tf.keras.layers.Flatten()(head_model)
head_model = tf.keras.layers.Dense(128, activation="relu")(head_model)
head_model = tf.keras.layers.Dropout(0.5)(head_model)
out_put = tf.keras.layers.Dense(2, activation="softmax")(head_model)

model = tf.keras.Model(inputs=base_model.input, outputs=out_put)

#freeze all the layers in base model
for layer in base_model.layers:
    layer.trainable = False

#compile our model
print("[INFO] compiling model...")
opt = tf.keras.optimizers.Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#train some head layer of model
print("[INFO] training model...")
H = model.fit(aug.flow(X_train, y_train, batch_size=BATCH_SIZE),
              validation_data=(X_test, y_test),
              steps_per_epoch=len(X_train)/BATCH_SIZE,
              epochs=EPOCHS)

#make predictions on the test set
print("[INFO] evaluating...")
prob = model.predict(X_test, batch_size=BATCH_SIZE)
pred = np.argmax(prob, axis=1)

#show classification report
print(classification_report(y_test.argmax(axis=1), pred, target_names=lb.classes_))

#seralize the model to disk
print("[INFO] saving mask detectot model...")
model.save(args["model"], save_format="h5")

#plot the traing loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["accuracy"], label="train_accuracy")
plt.plot(np.arange(0, EPOCHS), H.history["val_accuracy"], label="val_accuracy")
plt.title("Training Loss and Accuracy")
plt.xlabel("# Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])












