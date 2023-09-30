import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret,frame = cap.read()
    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    rect = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in rect:
        cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0))
        roi = frame[y:y+h,x:x+w]
        blur = cv.GaussianBlur(roi,(15,15),30)
        frame[y:y+blur.shape[0],x:x+blur.shape[1]]= blur
       
    cv.imshow('frame',frame)
    key = cv.waitKey(5)
    if key ==27:
        break
