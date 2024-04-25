import numpy as np
import cv2

image = cv2.imread('images/pomme.jpg',-1)

image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

cv2.namedWindow('Image HSV', cv2.WINDOW_NORMAL)

cv2.resizeWindow('Image HSV', 500, 400)


cv2.imshow('Image HSV', image_hsv)
cv2.waitKey(0)
cv2.destroyAllWindows()