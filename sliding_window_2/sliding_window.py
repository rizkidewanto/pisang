#Diresize gambarnya supaya bisa kecil

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

image = cv2.imread('pisang.jpg') # your image path
tmp = image # for drawing a rectangle
stepSize = 50
model=load_model('classifier_10.h5')
(w_width, w_height) = (50, 50) # window size
for x in range(0, image.shape[1] - w_width , stepSize):
    for y in range(0, image.shape[0] - w_height, stepSize):
        window = image[x:x + w_width, y:y + w_height, :]
        window=cv2.resize(window,dsize=(10,10))
        window = np.expand_dims(window, axis=0)
        prediksi = model.predict(window)
        prediksi = np.argmax(prediksi, axis=1)
        print(prediksi) #For Troubleshoot
        #cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (0, 0, 255), 2)
        #if prediksi==0:
        #    cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2)
        #elif prediksi==1:
        #    cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (0, 255, 0), 2)
        #elif prediksi==2:
        #    cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (0, 0, 255), 2)
        plt.imshow(np.array(tmp).astype('uint8'))
#plt.show()
cv2.imshow('Gambar',np.array(tmp).astype('uint8'))
cv2.waitKey(0)