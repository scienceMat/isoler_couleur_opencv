import os
import cv2
import numpy as np

# Set the current working directory
path = os.getcwd()

# Define input and output directories
inputdir = os.path.join(path, "images")
outPut_dir = os.path.join(path, 'output')

# Create the output directory if it doesn't exist
os.makedirs(outPut_dir, exist_ok=True)

# List all files in the input directory
files = os.listdir(inputdir)

for file in files:
    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
        fitem = os.path.join(inputdir, file)
        fout = os.path.join(outPut_dir, file)

        img = cv2.imread(fitem)
        if img is not None:
            # Convertir l'image en niveaux de gris et appliquer un seuillage
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # Trouver les contours
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # Créer une image noire pour dessiner les contours
            black_image = np.zeros_like(img)

            # Dessiner tous les contours
            cv2.drawContours(black_image, contours, -1, (0, 255, 0), 2)  # -1 signifie dessiner tous les contours

            # Sauvegarder l'image résultante
            cv2.imwrite(fout, black_image)
        else:
            print(f"Erreur de chargement de l'image: {fitem}")
    else:
        print(f"Format non supporté pour le fichier: {file}")