import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
import easyocr

# Chargement et prétraitement
image = cv2.imread('images/plaque.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Définition des seuils pour isoler le blanc (ajuster selon le besoin)
lower_white = np.array([0, 0, 150], dtype=np.uint8)
upper_white = np.array([180, 80, 255], dtype=np.uint8)
mask = cv2.inRange(hsv, lower_white, upper_white)

# Opérations morphologiques pour nettoyer le masque
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Trouver les contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Filtrage des contours par aspect ratio
final_contours = []
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / float(h)
    if 2 <= aspect_ratio <= 5:  # Aspect ratio typique pour les plaques
        final_contours.append(contour)
        break  # Supposons que la première correspondante est notre plaque



# Si aucun contour n'est adapté, sortir du script
if not final_contours:
    print("No suitable contour found")
    exit()

# Récupérer le premier contour valide
contour = final_contours[0]
x, y, w, h = cv2.boundingRect(contour)

# Dessiner le contour et le rectangle englobant sur l'image
cv2.drawContours(image, [contour], -1, (0, 255, 0), 3)
cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# Utiliser EasyOCR pour reconnaître le texte sur la plaque
reader = easyocr.Reader(['en'])  # Assurez-vous que 'en' est correct ou utilisez 'fr' pour le français
cropped_image = gray[y:y+h, x:x+w]

result = reader.readtext(cropped_image, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')

# Affichage des résultats de la détection de texte
if result:
    text = result[0][-2]
    cv2.putText(image, text, (x, y + h + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
else:
    print("No text detected")

# Afficher l'image finale
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Final Result with OCR")
plt.show()