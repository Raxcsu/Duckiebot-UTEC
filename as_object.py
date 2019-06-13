import cv2
import numpy as np


def nothing(x):
    pass


cap = cv2.VideoCapture("images/film5.avi")

while (True):
    ret, frame = cap.read()
    blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
    ret2, thresh = cv2.threshold(gray, 127, 255, 0)

    # First
    l_g = np.array([20, 100, 100])
    u_g = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, l_g, u_g)

    # Second

    l_w = np.array([ 20, 100, 100])
    u_w = np.array([20, 100, 100])

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)
        # Momentos
        M = cv2.moments(contour)
        # calculate x,y coordinate of center
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            print(cX)
        else:
            cX, cY = 0, 0
        cv2.circle(frame, (cX, cY), 5, (255, 255, 255), -1)
        cv2.imshow("normal", frame)
        cv2.imshow("video", mask)

    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
