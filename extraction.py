# -*- coding: utf-8 -*-

import numpy as np
import cv2


def extractface(image):
    roi=None
    face_cascade = cv2.CascadeClassifier('F:/Anaconda2/Library/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
    faces = face_cascade.detectMultiScale(image, 1.3, 5)
    for (x,y,w,h) in faces:
              roi=image[y:y+h, x:x+w]
    
    if(roi is None):
        return image
    else :
        return roi



