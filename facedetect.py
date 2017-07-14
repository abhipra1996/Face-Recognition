import numpy as np
import cv2
import time
from lbp import LBP
from matplotlib import pyplot as plt
from sklearn import svm
import scipy.signal
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

Test_X=[]
size=256
Test_Y=0
count=0.0
    

Train_X=np.load("Training_XMatrix256_64.dat")
Train_Y=np.load("Training_YMatrix256_64.dat")
Naming_labels=np.load("Name_labels_256_64.dat")
classifier = SVC(kernel='linear')
classifier.fit(Train_X,Train_Y)
lag=[]
frames=0


face_cascade = cv2.CascadeClassifier('F:/Anaconda2/Library/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')


camera = cv2.VideoCapture(0)
time.sleep(2)

while(True):
    ret, frame = camera.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    starttime=time.time()
    for (x,y,w,h) in faces:
        
	        face=gray[y: y + h, x: x + w]
	        
	        face=cv2.resize(face,(256,256))
	        scipy.signal.medfilt(face)
	        pre=LBP(face)
	        Test_X=pre       
	        Test_X=np.array(Test_X).reshape(1,-1)
	        Test_X.transpose()
	        prediction=classifier.predict(Test_X)	       
	        endtime=time.time()
	        timetaken=endtime-starttime
	        lag.append(timetaken)    
	        predictedname=Naming_labels[prediction]    
	        for i in predictedname:
                    pname = i;
	        pname=pname+str(timetaken)
	        frames=frames+1
	       
	        font = cv2.FONT_HERSHEY_SIMPLEX
	        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
	        cv2.putText(frame,pname,(x,y), font, 1,(255,255,255),2)
	       
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
   
    
camera.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

lag=np.array(lag)
meandelay=np.mean(lag)
print 'Average Delay Time = ', meandelay
print 'Total Number Of Frames = ' ,frames


