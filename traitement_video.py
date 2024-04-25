import numpy as np
import cv2

# Ouvrir le flux vidéo
cap = cv2.VideoCapture('images\pomme_video.mp4')

# Vérifier si la vidéo a été chargée correctement
if not cap.isOpened():
    print("Erreur : Impossible d'ouvrir le flux vidéo")
    exit()

# Boucle de lecture vidéo
while True:
    ret, frame = cap.read()  # lire une frame de la vidéo
    if not ret:
        print("Erreur : Impossible de lire la vidéo ou fin de la vidéo atteinte")
        break

    # Convertir la frame de BGR à HSV
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Définir la plage de la couleur rouge en HSV
    lower_red = np.array([168, 25, 25])
    upper_red = np.array([180, 255, 255])

    # Créer un masque pour isoler les parties rouges de la frame
    mask_red = cv2.inRange(frame_hsv, lower_red, upper_red)

    # Appliquer le masque à la frame
    result_red = cv2.bitwise_and(frame, frame, mask=mask_red)

    # Afficher le résultat
    cv2.imshow('Frame traitée', result_red)

    # Arrêter la lecture si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer le flux vidéo et fermer toutes les fenêtres
cap.release()
cv2.destroyAllWindows()