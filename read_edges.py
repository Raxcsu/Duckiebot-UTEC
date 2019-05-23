import cv2
import numpy as np
import matplotlib.pyplot as plt

vd = cv2.VideoCapture("images/film5.avi")
im = cv2.imread("images/test_image.jpg")


def toCanny(src):
    gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)  # Convert to gray
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)  # Detect edges
    return canny


def selectArea(src):
    polygon = np.array([[(5, 480), (270, 156), (410, 164), (640, 340), (640, 480)]])
    mask = np.zeros_like(src)
    cv2.fillPoly(mask, polygon, 255)
    mask_img = cv2.bitwise_and(src, mask)
    return mask_img

def selectEdge(src):
    polygon = np.array([[(380, 480), (380, 155), (388, 158), (640, 446), (640, 480)]])
    mask = np.zeros_like(src)
    cv2.fillPoly(mask, polygon, 255)
    mask_img = cv2.bitwise_and(src, mask)
    return mask_img

def disLines(src, lines):
    line_image = np.zeros_like(src)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return line_image


while (vd.isOpened()):

    ret, frame = vd.read()
    l_im = np.copy(frame)
    # Our operations on the frame come here
    canny = toCanny(frame)
    # Select image
    selectImg = selectArea(canny)
    selectEd = selectEdge(canny)

    contours, hierarchy = cv2.findContours(selectImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for contour in contours:
        cv2.drawContours(frame, contour, -1, (0, 255, 0), 3)

    # Detect lines
    line = cv2.HoughLinesP(selectImg, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    line_image = disLines(l_im, line)

    # Union

    union_img = cv2.addWeighted(l_im, 0.8, line_image, 1, 1)

    # Display the resulting frame
#    plt.imshow(frame)
#    if plt.show(1) & 0xFF == ord('q'):
#        break

    cv2.imshow("wd", union_img)
    cv2.imshow("wd2", frame)
    key = cv2.waitKey(10)
    if key == 5:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vd.release()
cv2.destroyAllWindows()
