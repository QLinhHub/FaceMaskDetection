#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:35:29 2020

@author: quanglinhle
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import argparse
import os
import numpy as np
import cv2

#construct argument parser and parse argument
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path to input image")
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float,
                default=0.5,
                help="minimum probability to filter weak detection")
args = vars(ap.parse_args())

#load serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

#load trained face detector model from disk
print("[INFO] loading face mask detector model...")
model = load_model(args["model"])

#load the input image from disk
image = cv2.imread(args["image"])
orig_img = image.copy()
(h, w) = image.shape[:2]

#construct a blob from image
blob = cv2.dnn.blobFromImage(image, 1.0, (300,300), (104.0, 177.0, 123.0))

#pass the blob through the network and obtain the face detection
print("[INFO] detecting face...")
net.setInput(blob)
detections = net.forward() # detections have shape (1x1xNx7) 
                            #where N is the number of bouding boxes found and each row 
                            #contains a 7-dimensions vectors:
                            #([batchId, classId, confidence, left, top, right, bottom])

#loop over the detections
for i in range(0, detections.shape[2]):
    confidence = detections[0,0,i,2]
    if confidence > args["confidence"]:
        #compute the coordinate of bouding box
        box = detections[0,0,i,3:7] * np.array([w,h,w,h])
        (x_start, y_start, x_end, y_end) = box.astype("int")
        
        #ensure the bouding box fall within the dimensions of the frame 
        (x_start, y_start) = (max(0, x_start), max(0, y_start))
        (x_end, y_end) = (min(w-1, x_end), min(h-1, y_end))

        #extract the face ROI, convert it to RGB ordering, resize it to 224x224 and preprocess it
        face = image[y_start:y_end, x_start:x_end]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224,224))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        #pass the face through the model to determine if the face has the mask or not
        (mask_prob, non_mask_prob) = model.predict(face)[0]
        
        #determine class label and text, bouding box color
        label = "Mask" if mask_prob > non_mask_prob else "No Mask"
        color = (0,255,0) if label=="Mask" else (0,0,255)
        
        #include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask_prob, non_mask_prob)*100)
        
        #display the bouding box and label on the output frame
        cv2.rectangle(image, (x_start, y_start), (x_end, y_end), color, 2)
        cv2.putText(image, label, (x_start, y_start-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
#show the ouput image
cv2.imshow("Output result", image)
cv2.waitKey(0)














