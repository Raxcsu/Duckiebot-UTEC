import cv2
import numpy as np

image = "images/road1.jpeg"

cap = cv2.imread(image)
def nothing(x):
    pass

cv2.namedWindow('result')

h,s,v = 100,100,100

# track bar
cv2.createTrackbar('h', 'result',0,179,nothing)
cv2.createTrackbar('s', 'result',0,255,nothing)
cv2.createTrackbar('v', 'result',0,255,nothing)

while(1):

    cap = cv2.imread(image)

    #conversion
    hsv = cv2.cvtColor(cap,cv2.COLOR_BGR2HSV)

    h = cv2.getTrackbarPos('h','result')
    s = cv2.getTrackbarPos('s','result')
    v = cv2.getTrackbarPos('v','result')

    # mascara
    lower = np.array([h,s,v])
    upper = np.array([180,255,255])

    mask = cv2.inRange(hsv,lower, upper)

    mask_image = cv2.bitwise_and(cap,cap,mask = mask)

    cv2.imshow('result',mask_image)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()

cv2.destroyAllWindows()
