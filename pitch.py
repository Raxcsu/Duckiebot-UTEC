import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

filename = "images/testa37.png"
filename2 = "images/camera_cal/calibration1.png"
cap = cv2.VideoCapture("images/film5.avi")
w = 640
img = cv2.imread(filename, 0)
img2 = cv2.imread(filename2,0)

# Matriz y coeficientes de calibración basadas en otras cámaras

mtrz = np.float32([
    [297.44663835 , 0.00000000e+00, 321.20418466],
    [0.00000000e+00, 296.91848716, 216.10159361],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])
dist = np.float32([
    [0.03942281,  0.08875919, -0.00474551, -0.00113336, -0.21502914]
])

# Puntos de la imagen inicial
src = np.float32([
    [55, 480],
    [295, 173],
    [479, 190],
    [640, 320]
    #[640, 480]
])
# Puntos de la nueva imagen
dst = np.float32([
    [180, 480],
    [180, 0],
    [w - 180, 0],
    #[w - 180, 215],
    [w - 180, 480]
])


def sinDistorision (img, mtrz, distCoff):
    undist = cv2.undistort(img, mtrz, distCoff, None, mtrz)
    return undist


def drawPoints (img, src):
    result = np.copy(img)
    color = [255, 0, 0] # Red
    espesor = -1
    radio = 5
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]
    x3, y3 = src[3]
#    x4, y4 = src[4]
    cv2.circle(result, (x0, y0), radio, color, espesor)
    cv2.circle(result, (x1, y1), radio, color, espesor)
    cv2.circle(result, (x2, y2), radio, color, espesor)
    cv2.circle(result, (x3, y3), radio, color, espesor)
#    cv2.circle(result, (x4, y4), radio, color, espesor)
    return result

def drawLines (img, src):
    result = np.copy(img)
    color = [255, 0, 0] # Red
    espesor = 2
    x0, y0 = src[0]
    x1, y1 = src[1]
    x2, y2 = src[2]
    x3, y3 = src[3]
#    x4, y4 = src[4]
    cv2.line(result, (x0, y0), (x1, y1), color, espesor)
    cv2.line(result, (x1, y1), (x2, y2), color, espesor)
    cv2.line(result, (x2, y2), (x3, y3), color, espesor)
    cv2.line(result, (x3, y3), (x0, y0), color, espesor)
#    cv2.line(result, (x4, y4), (x0, y0), color, espesor)
    return result

def deformado(img):
    #Transformacion perspectiva
    img_size = (img.shape[1], img.shape[0])
    trans = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, trans, img_size, flags=cv2.INTER_NEAREST)
    return warped


def noDeformado(img):
    #Transformación inversa
    img_size = (img.shape[1], img.shape[0])
    trans = cv2.getPerspectiveTransform(dst, src)
    unwarped = cv2.warpPerspective(img, trans, img_size, flags=cv2.INTER_NEAREST)
    return unwarped

while (True):
    undist = sinDistorision(img, mtrz, dist)
    cv2.imshow("normal", drawLines(undist, src))
    #Vista Aerea
    imgSrc = np.zeros_like(deformado(img))
    imgDst = np.zeros_like(deformado(img))
    src_points_img = drawPoints(imgSrc, src)
    src_points_img = drawLines(src_points_img, src)
    dst_points_warped = drawPoints(imgDst, dst)
    dst_points_warped = drawLines(dst_points_warped, dst)
    cv2.imshow("SP", src_points_img)
    cv2.imshow("WP", dst_points_warped)
    cv2.imshow("",img)

    images = glob.glob('images/testa*.png')

    for fname in images:
        img = mpimg.imread(fname)

        # Undistort the image based on the camera calibration
        undist = sinDistorision(img, mtrz, dist)

        # warp the image
        warped = deformado(undist)

        # add the points to the og and warped images
        src_points_img = drawPoints(img, src)
        src_points_img = drawLines(src_points_img, src)
        dst_points_warped = drawPoints(warped, dst)
        dst_points_warped = drawLines(dst_points_warped, dst)
        cv2.imshow("SP2", src_points_img)
        cv2.imshow("WP2", dst_points_warped)

    key = cv2.waitKey(2)
    if key == 27:
        break
cap.release()
cv2.destroyAllWindows()
