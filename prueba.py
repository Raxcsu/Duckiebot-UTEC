import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = "images/road1.jpeg"
filevideo = "images/film5.avi"

img = cv2.imread(filename,0)
cap = cv2.VideoCapture(filevideo)   #el index es la camara que seleccionas de las ya existentes

while(True):

    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
'''
k = cv2.waitKey(0) & 0xFF
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
'''
#Show in Matplotlib
plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#plt.show()