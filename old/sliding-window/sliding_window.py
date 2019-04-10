# USAGE
# python sliding_window.py --image <gambar>.jpg 

# import the necessary packages
from sliding.helpers import pyramid
from sliding.helpers import sliding_window
from sliding.nms import non_max_suppression_fast
from keras.models import load_model
import keras.preprocessing.image as gbr
import numpy as np
import h5py
import argparse
import time
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

# load the image and define the window width and height
image = cv2.imread(args["image"])
(winW, winH) = (150, 150) #(128, 128)
matang=[]
mentah=[]
terlalu=[]

model=load_model('classifier.h5')

# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize=32, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        crop = image[y:y+winH, x:x+winW]
        crop = np.expand_dims(crop, axis=0)
        prediksi = model.predict(crop)
        prediksi = np.argmax(prediksi, axis=1)
        
        clone = resized.copy()
        #Untuk matang
        if prediksi==0:
            #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (255, 0, 0), 2) #(0,255,0) -> B,G,R
            matang.append([x,y,winW,winH])
        #Untuk mentah
        elif prediksi==1:
            #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
            mentah.append([x,y,winW,winH])
        #Untuk terlalu matang
        elif prediksi==2:
            #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 0, 255), 2)
            terlalu.append([x,y,winW,winH])
            
matang=np.array(matang)
mentah=np.array(mentah)
terlalu=np.array(terlalu)
		#cv2.imshow("Window", clone)
		#cv2.waitKey(1)
		#time.sleep(0.025)
        
        #====Apply Non Max Supression here====#
kotak = non_max_suppression_fast(matang, 0.3)
for (startX, startY, endX, endY) in kotak:
    cv2.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)
kotak = non_max_suppression_fast(mentah, 0.3)
for (startX, startY, endX, endY) in kotak:
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
kotak = non_max_suppression_fast(terlalu, 0.3)
for (startX, startY, endX, endY) in kotak:
    cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)

cv2.imshow("Hasil", image)
cv2.waitKey(0)