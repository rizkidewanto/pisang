import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model 

image = cv2.imread("pisang.jpg")
tmp = image
stepSize = 50
model=load_model('classifier_50.h5')
(w_width, w_height) = (50, 50) # window size
for x in range(0, image.shape[1] - w_width , stepSize):
    for y in range(0, image.shape[0] - w_height, stepSize):
        print("Nilai x : ",x)
        print("Nilai y : ",y)
        window = image[x:x + w_width, y:y + w_height, :]
        
        #window = np.expand_dims(window, axis=0)
        #prediksi = model.predict(window)
        #prediksi = np.argmax(prediksi, axis=1)
        #print(prediksi) #For Troubleshoot
        
        cv2.rectangle(tmp, (x, y), (x + w_width, y + w_height), (255, 0, 0), 2) # draw rectangle on image
        plt.imshow(np.array(tmp).astype('uint8'))
# show all windows
plt.show()

