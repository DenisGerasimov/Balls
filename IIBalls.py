import cv2
import numpy as np
import time

cam = cv2.VideoCapture(0)
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

lower = (3, 150, 200)
upper = (30, 255, 255)

#blue
#lower = (70, 90, 200)
#upper = (110, 255, 255)

#red
#lower = (0, 160, 150)
#upper = (2, 255, 255)

while cam.isOpened():
    _, image = cam.read()
    curr_time = time.time()
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    ret, fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
    fg = np.uint8(fg)
    confuse = cv2.subtract(mask, fg)
    ret, markers = cv2.connectedComponents(fg)
    markers += 1
    markers[confuse == 255] = 0

    wmarkers = cv2.watershed(image, markers.copy())
    contours, hierarchy = cv2.findContours(wmarkers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    for i in range(len(contours)):
        if hierarchy[0][i][3] == -1:
            cv2.drawContours(image, contours, i, (0, 255, 0), 6)

    print(int((len(hierarchy[0]) -1 )/ 2))

    cv2.imshow("Camera", image)
    cv2.imshow("Mask", mask)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
