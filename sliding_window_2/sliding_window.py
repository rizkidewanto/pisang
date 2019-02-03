import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

# read the image and define the stepSize and window size 
# (width,height)
image = cv2.imread('pisang.jpg') # your image path
tmp = image # for drawing a rectangle
stepSize = 150
model=load_model('classifier.h5')
(w_width, w_height) = (150, 150) # window size
for x in range(0, image.shape[1] - w_width , stepSize):
    for y in range(0, image.shape[0] - w_height, stepSize):
        window = image[x:x + w_width, y:y + w_height, :]

        # classify content of the window with your classifier and 
        # determine if the window includes an object (cell) or not
        window = np.expand_dims(window, axis=0)
        prediksi = model.predict(window)
        prediksi = np.argmax(prediksi, axis=1)
        
        if prediksi==0:
            cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2)
        elif prediksi==1:
            cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (0, 255, 0), 2)
        elif prediksi==2:
            cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (0, 0, 255), 2)
        plt.imshow(np.array(tmp).astype('uint8'))
# show all windows
#plt.show()
cv2.imshow('Gambar',np.array(tmp).astype('uint8'))
cv2.waitKey(0)