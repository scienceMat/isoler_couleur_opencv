import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Charger l'image en niveaux de gris
image = cv2.imread('images/flechette.jpg', cv2.IMREAD_GRAYSCALE)
histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).ravel()
total_pixels = histogram.sum()

# Fonction pour calculer la variance intra-classe
def calculate_variance(threshold):
    background = histogram[:threshold]
    object = histogram[threshold:]

    weight_background = background.sum() / total_pixels
    weight_object = object.sum() / total_pixels

    if background.sum() == 0 or object.sum() == 0:
        return np.nan

    mean_background = np.sum(background * np.arange(threshold)) / background.sum()
    mean_object = np.sum(object * np.arange(threshold, 256)) / object.sum()

    variance_background = np.sum(((np.arange(threshold) - mean_background) ** 2) * background) / background.sum()
    variance_object = np.sum(((np.arange(threshold, 256) - mean_object) ** 2) * object) / object.sum()

    return weight_background * variance_background + weight_object * variance_object

# Initialiser le plot
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
threshold = 128
variance = calculate_variance(threshold)
line, = plt.plot(histogram, label='Histogram')
threshold_line = plt.axvline(x=threshold, color='r', label='Threshold')
plt.title(f'Variance Intra-Class: {variance:.2f}')
plt.legend()

# Ajouter un Slider
axcolor = 'lightgoldenrodyellow'
ax_thresh = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
thresh_slider = Slider(ax=ax_thresh, label='Threshold', valmin=0, valmax=255, valinit=threshold)

# Fonction de mise à jour pour le Slider
def update(val):
    new_thresh = int(thresh_slider.val)
    new_variance = calculate_variance(new_thresh)
    threshold_line.set_xdata([new_thresh, new_thresh])
    plt.title(f'Variance Intra-Class: {new_variance:.2f}')
    fig.canvas.draw_idle()

thresh_slider.on_changed(update)

plt.show()