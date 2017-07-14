import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
from lbp import LBP
import scipy.signal
from extraction import extractface
from sklearn.decomposition import PCA

size=256

base_path_training = 'C:/Users/dell pc/Desktop/database' 
Train_X=[]
Train_Y=[]
Naming_labels=[]
j=0;
#count=1;

for folders in os.listdir(base_path_training):
	folder=base_path_training+'/'+folders
	print j, " " ,folders
	Naming_labels.append(folders)
	k=0
	for images in os.listdir(folder):
		image=folder+'/'+images
		#print 'path=',image
		imggrey = cv2.imread(image,0)
		scipy.signal.medfilt(imggrey)
		extracted_face=extractface(imggrey)
		k=k+1
		print k
		pre=LBP(extracted_face)
		Train_X.append(pre)
		Train_Y.append(j)
	j=j+1

Naming_labels=np.array(Naming_labels)
Naming_labels.dump("Name_labels_256_5_test.dat")
Train_X=np.array(Train_X)
Train_X.transpose()
Train_Y=np.array(Train_Y)
print Train_X.shape
print Train_Y.shape
print 'Training Done'
print Train_Y

Train_X.dump("Training_XMatrix256_5_test.dat")
Train_Y.dump("Training_YMatrix256_5_test.dat")


 