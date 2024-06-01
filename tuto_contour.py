import numpy as np
import cv2

image = cv2.imread('images/flechette.jpg',-1)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


contours, _ = cv2.findContours(th2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Créer une image noire pour dessiner les contours
black_image = np.zeros_like(image)

# Dessiner tous les contours
cv2.drawContours(black_image, contours, -1, (0, 255, 0), 2)  # -1 signifie dessiner tous les contours


cv2.namedWindow('Image', cv2.WINDOW_NORMAL)

cv2.resizeWindow('Image', 500, 400)

cv2.imshow('Image', black_image)
cv2.waitKey(0)
cv2.destroyAllWindows()