# -*- coding: utf-8 -*-

import numpy as np
import cv2

size=256


def LBP(img): 
    
    def thresholded(center, pixels):
      out = []
      for a in pixels:
        if a >= center:
            out.append(1)
        else:
            out.append(0)
      return out

    
    def get_pixel_else_0(img,x,y):
      size=len(img)
      default=0
      if x<0 or y<0 or x>size-1 or y>size-1 :
        return default
      else:
        return img[x][y]




    image=cv2.resize(img,(size,size))
    temp=np.zeros((size,size),dtype=np.uint8)
    for x in range(0, len(image)):
    		 for y in range(0, len(image[0])):
			  
			   center = image[x,y]
			  
			   top_left      = get_pixel_else_0(image, x-1, y-1)
			  
			   top_up        = get_pixel_else_0(image, x-1, y)
			
			   top_right     = get_pixel_else_0(image, x-1, y+1)
			
			   right         = get_pixel_else_0(image, x, y+1 )
			  
			   left          = get_pixel_else_0(image, x, y-1 )
			  
			   bottom_left   = get_pixel_else_0(image, x+1, y-1)
			   
			   bottom_right  = get_pixel_else_0(image, x+1, y+1)
			  
			   bottom_down   = get_pixel_else_0(image, x+1,   y )
			   
			   values=[]
			   values = thresholded(center, [top_left, top_up, top_right, right, bottom_right,
				                      bottom_down, bottom_left, left])
			 
			   weights = [1, 2 , 4, 8 , 16 , 32, 64, 128]
			   res = 0
			   for a in range(0, len(values)):
			       res += weights[a] * values[a]
             
			   temp[x][y]=res
    
    hist, bin_edges = np.histogram(temp,bins=64)
    
    return hist
