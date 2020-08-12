#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 09:36:07 2020

@author: quanglinhle
"""

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import os
import time
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
                default="face_detector",
                help="path to the face detector directory")
ap.add_argument("-m", "--model", type=str,
                default="mask_detector.model",
                help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104.0, 177.0, 123.0))
    
    #pass the blob through the faceNet
    faceNet.setInput(blob)
    detections = faceNet.forward()
        
    faces = []
    locs = []
    preds = []
    
    #loop over the detections and compute the coordinates of bounding boxes
    for i in range(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > args["confidence"]:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (x_start, y_start, x_end, y_end) = box.astype("int")
            
            #ensure the coordinate fall within the frame
            (x_start, y_start) = (max(0, x_start), max(0, y_start))
            (x_end, y_end) = (min(w-1, x_end), min(h-1, y_end))
            
            #extract the face from the frame and make prediction using maskNet
            face = frame[y_start:y_end, x_start:x_end]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224,224))
            face = img_to_array(face)
            face = preprocess_input(face)
            
            faces.append(face)
            locs.append((x_start, y_start, x_end, y_end))
        
    #make predictions on batch(all faces detected in frame)
    #note: in detect_mask_image.py, we make prediction for every loop 
    if len(faces) > 0:
        faces = np.array(faces, dtype=np.float32)
        preds = maskNet.predict(faces)
        
    return (locs, preds)

#load face detector model
print("INFO loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#load face mask detector model
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

#initialize video stream and allow camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#loop over the frames from the video stream
while True:
    #grab the frames from the threaded video stream and resize its maximum width to 400pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    #detect face and detect mask
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
    
    #loop over (locs, preds)
    for (box, pred) in zip(locs, preds):
        (x_start, y_start, x_end, y_end) = box
        (mask_prob, non_mask_prob) = pred
        
        #determine the class lable and color to draw bouding box and text
        label="Mask" if mask_prob > non_mask_prob else "No Mask"
        color=[0,255,0] if label=="Mask" else [0,0,255]
        
        label = "{}: {:.2f}%".format(label, max(mask_prob, non_mask_prob)*100)
        
        #display bouding box and text on the output frame
        cv2.putText(frame, label, (x_start, y_start-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), color, 2)
        
    #show the output frame
    cv2.imshow("Output result", frame)
    key = cv2.waitKey(1) & 0xFF 
    #break if press "q"    
    if key == ord("q"):
        break
    
#do a bit of clean up
cv2.destroyAllWindows()
vs.stop()
        
        











            
            
            
            
            
            
            
            
            
            
            
            
            