import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((7 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane.

# Make a list of calibration images
fname = 'images/camera_cal/calibration1.png'

fig, axs = plt.subplots(5, 4, figsize=(16, 11))
fig.subplots_adjust(hspace=.2, wspace=.001)
axs = axs.ravel()

# Step through the list of images and search for chessboard corners in each one

img = cv2.imread(fname)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Find the chessboard corners
ret, corners = cv2.findChessboardCorners(gray, (9, 7), None)

# If found, add object points, image points to the lists
if ret == True:
    objpoints.append(objp)
    imgpoints.append(corners)

    # Draw and display the corners
    img = cv2.drawChessboardCorners(img, (9, 7), corners, ret)

    cv2.imshow("",img)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print("Camera Matrix")
    print(mtx)
    print("Distortion Coeffs")
    print(dist)