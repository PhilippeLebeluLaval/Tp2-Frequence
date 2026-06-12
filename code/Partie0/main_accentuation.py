"""
main_accentuation.py

TP - Accentuation d'images (Unsharp Masking)

Ce script:
1. Charge deux images
2. Applique une accentuation (sharpening) via unsharp masking
3. Affiche les résultats

Auteur: [Votre nom]
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.util import img_as_float
from skimage.filters import gaussian


def unsharp_mask(image, sigma=2, alpha=1.5):
    """
    Applique la technique de Unsharp Masking.

    Paramètres
    ----------
    image : ndarray
        Image d'entrée (float entre 0 et 1)
    sigma : float
        Écart-type du filtre gaussien (contrôle le flou)
    alpha : float
        Facteur d'accentuation

    Retour
    ------
    sharpened : ndarray
        Image accentuée
    """

    # 1. Floutage de l'image
    blurred = gaussian(image, sigma=sigma)

    # 2. Extraction des détails (hautes fréquences)
    detail = image - blurred

    # 3. Ajout amplifié des détails à l'image originale
    sharpened = image + alpha * detail

    # 4. Clipping pour rester dans l'intervalle valide [0,1]
    sharpened = np.clip(sharpened, 0, 1)

    return sharpened


def main():
    """
    Fonction principale.
    """

    # -------------------------
    # 1. Chargement des images
    # -------------------------
    image1 = img_as_float(io.imread("image1.jpg"))
    image2 = img_as_float(io.imread("image2.jpg"))

    # -------------------------
    # 2. Accentuation
    # -------------------------
    sharpened1 = unsharp_mask(image1, sigma=2, alpha=1.5)
    sharpened2 = unsharp_mask(image2, sigma=3, alpha=2.0)

    # -------------------------
    # 3. Affichage des résultats
    # -------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].imshow(image1)
    axes[0, 0].set_title("Image 1 - Originale")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(sharpened1)
    axes[0, 1].set_title("Image 1 - Accentuee")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(image2)
    axes[1, 0].set_title("Image 2 - Originale")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(sharpened2)
    axes[1, 1].set_title("Image 2 - Accentuee")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
