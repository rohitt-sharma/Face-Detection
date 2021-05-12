# Face-Detection
Face Detection using OpenCv
import cv2
import numpy as np

video = cv2.VideoCapture(2) # if single webcam 0, or else ip number of webcam
faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

# Using Viola & Jones
# we detect the face and put bounding box around it

while True:
    _,img = video.read()  # returns 2 variable, 1st was the flag to see if the fram was red correctly, 2nd was frame itself

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1,10)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,0),2)

    cv2.imshow('img', img)

    k = cv2.waitKey(30) & 0xff  #
    if k == 27:  # this tells to break the loop if escape key is pressed
        break

# cap.release()
