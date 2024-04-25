import numpy as np
import cv2

image = cv2.imread('images/nevermind.png',-1)

cv2.imshow('Image avec ligne', image)
cv2.waitKey(0)
cv2.destroyAllWindows()