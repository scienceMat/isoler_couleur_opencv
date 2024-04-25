import numpy as np
import cv2

image = cv2.imread('images/pomme.jpg',-1)

image_hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

lower_red = np.array([168,25,25])
upper_red  = np.array([180,255,255])

mask = cv2.inRange(image_hsv, lower_red, upper_red)

cv2.namedWindow('Image HSV', cv2.WINDOW_NORMAL)

cv2.resizeWindow('Image HSV', 500, 400)


cv2.imshow('Image HSV', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()