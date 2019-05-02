import cv2
import numpy as np
video = cv2.VideoCapture("pista1.mp4")

distResolution = 1
angleResolution = 1*np.pi/180
minVotes = 5
minLineLength = 5
maxLineGap = 10

while True:
    
    ret, orig_frame = video.read()

    orig_frame = cv2.flip(orig_frame, -1)

    if not ret:
        video = cv2.VideoCapture("pista1.mp4")
        continue
   
    frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    low_yellow = np.array([20, 42, 195])
    up_yellow = np.array([54, 255, 255])
    mask = cv2.inRange(hsv, low_yellow, up_yellow)
    
    edges = cv2.Canny(mask, 100, 150)
    lines = cv2.HoughLinesP(edges, distResolution, angleResolution, minVotes, maxLineGap= maxLineGap)
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    cv2.imshow("mask",  frame)
    cv2.imshow("edges", edges)
    
    key = cv2.waitKey(3)
    if key == 5:
        break

video.release()
cv2.destroyAllWindows()