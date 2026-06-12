import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter
import os
import sys

# Importer les fonctions de piles depuis main_pile.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from main_pile import gaussian_stack, laplacian_stack
    print("✓ Fonctions de piles importées depuis main_pile.py")
except ImportError:
    print("⚠️  Impossible d'importer depuis main_pile.py")
    print("    Définition locale des fonctions de piles...")

    def gaussian_stack(image, num_levels=5, sigma_base=2):
        """
        Crée une pile gaussienne en appliquant des filtres gaussiens
        de sigma croissant (doublé à chaque niveau) à l'image originale.
        """
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        else:
            image = image.astype(np.float32)

        if image.ndim == 2:
            stack = np.zeros((num_levels, *image.shape), dtype=np.float32)
        else:
            stack = np.zeros((num_levels, *image.shape), dtype=np.float32)

        sigmas = []

        for level in range(num_levels):
            sigma = sigma_base * (2 ** level)
            sigmas.append(sigma)

            if image.ndim == 3:
                filtered = gaussian_filter(image, sigma=(sigma, sigma, 0))
            else:
                filtered = gaussian_filter(image, sigma=sigma)

            stack[level] = filtered

        return stack, sigmas

    def laplacian_stack(gaussian_stack_array):
        """
        Crée une pile laplacienne à partir d'une pile gaussienne.
        """
        num_levels = gaussian_stack_array.shape[0]
        laplacian_stack_array = np.zeros_like(gaussian_stack_array)

        for level in range(num_levels - 1):
            laplacian_stack_array[level] = gaussian_stack_array[level] - gaussian_stack_array[level + 1]

        laplacian_stack_array[-1] = gaussian_stack_array[-1]

        return laplacian_stack_array


# -------------------------------------------------
# MÉLANGE MULTIRÉSOLUTION
# -------------------------------------------------

def multiresolution_blend(image1, image2, mask, num_levels=5):
    """
    Mélange deux images en utilisant la technique de mélange multirésolution
    basée sur les piles laplaciennes (Burt & Adelson, 1983).

    Parameters
    ----------
    image1 : ndarray
        Première image (RGB ou grayscale)
    image2 : ndarray
        Deuxième image (même taille que image1)
    mask : ndarray
        Masque binaire (0 ou 1) de même taille que les images
        1 = prendre image1, 0 = prendre image2
    num_levels : int
        Nombre de niveaux dans les piles

    Returns
    -------
    blended : ndarray
        Image mélangée
    blend_stack : ndarray
        Pile laplacienne du résultat (pour visualisation)
    """

    # Normaliser les images en float [0,1]
    if image1.dtype == np.uint8:
        image1 = image1.astype(np.float32) / 255.0
    else:
        image1 = image1.astype(np.float32)

    if image2.dtype == np.uint8:
        image2 = image2.astype(np.float32) / 255.0
    else:
        image2 = image2.astype(np.float32)

    if mask.dtype == np.uint8:
        mask = mask.astype(np.float32) / 255.0
    else:
        mask = mask.astype(np.float32)

    # Étape 1: Créer les piles laplaciennes pour les deux images
    print("  Création pile gaussienne image 1...")
    g_stack1, sigmas = gaussian_stack(image1, num_levels=num_levels)
    print("  Création pile laplacienne image 1...")
    l_stack1 = laplacian_stack(g_stack1)

    print("  Création pile gaussienne image 2...")
    g_stack2, _ = gaussian_stack(image2, num_levels=num_levels)
    print("  Création pile laplacienne image 2...")
    l_stack2 = laplacian_stack(g_stack2)

    # Étape 2: Créer la pile gaussienne pour le masque
    print("  Création pile gaussienne pour le masque...")
    g_mask_stack, _ = gaussian_stack(mask, num_levels=num_levels)

    # Étape 3: Mélanger les piles laplaciennes niveau par niveau
    # À chaque niveau: blend[i] = l1[i] * mask[i] + l2[i] * (1 - mask[i])
    print("  Mélange des piles laplaciennes...")
    blend_stack = np.zeros_like(l_stack1)

    for level in range(num_levels):
        mask_level = g_mask_stack[level]

        # Étendre le masque aux dimensions RGB si nécessaire
        if image1.ndim == 3 and mask_level.ndim == 2:
            mask_level = mask_level[:, :, np.newaxis]

        blend_stack[level] = l_stack1[level] * mask_level + l_stack2[level] * (1 - mask_level)

    # Étape 4: Reconstruire l'image finale en sommant tous les niveaux
    print("  Reconstruction de l'image finale...")
    blended = np.sum(blend_stack, axis=0)

    # Clipper dans [0, 1]
    blended = np.clip(blended, 0, 1)

    return blended, blend_stack, sigmas


# -------------------------------------------------
# CRÉATION DE MASQUES
# -------------------------------------------------

def create_vertical_mask(height, width, split_ratio=0.5):
    """
    Crée un masque vertical binaire.

    Parameters
    ----------
    height : int
        Hauteur de l'image
    width : int
        Largeur de l'image
    split_ratio : float
        Position de la séparation (0.5 = milieu)

    Returns
    -------
    mask : ndarray
        Masque binaire (1 à gauche, 0 à droite)
    """

    mask = np.zeros((height, width), dtype=np.float32)
    split_col = int(width * split_ratio)
    mask[:, :split_col] = 1.0

    return mask


def create_horizontal_mask(height, width, split_ratio=0.5, smoothness=50):
    """
    Crée un masque horizontal avec transition douce.
    Utilisé pour séparer le ciel (haut) de l'océan (bas).

    Parameters
    ----------
    height : int
        Hauteur de l'image
    width : int
        Largeur de l'image
    split_ratio : float
        Position de la ligne d'horizon (0.5 = milieu, 0.3 = plus bas)
    smoothness : float
        Lissage de la transition (sigma du flou gaussien)

    Returns
    -------
    mask : ndarray
        Masque avec transitions douces (1 en haut, 0 en bas)
    """

    mask = np.zeros((height, width), dtype=np.float32)
    split_row = int(height * split_ratio)

    # 1 en haut (ciel), 0 en bas (océan)
    mask[:split_row, :] = 1.0

    # Lisser la transition
    mask = gaussian_filter(mask, sigma=smoothness)

    # Normaliser entre 0 et 1
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    return mask


def create_irregular_mask(height, width, center_x_ratio=0.5, center_y_ratio=0.5,
                         radius_ratio=0.3, smoothness=50):
    """
    Crée un masque irrégulier circulaire/elliptique pour un mélange plus naturel.

    Parameters
    ----------
    height : int
        Hauteur de l'image
    width : int
        Largeur de l'image
    center_x_ratio : float
        Position X du centre (0-1)
    center_y_ratio : float
        Position Y du centre (0-1)
    radius_ratio : float
        Rayon relatif à la taille de l'image (0-1)
    smoothness : float
        Lissage du contour (sigma du flou gaussien)

    Returns
    -------
    mask : ndarray
        Masque avec transitions douces (valeurs continues 0-1)
    """

    # Créer une grille de coordonnées
    y, x = np.ogrid[:height, :width]
    center_x = int(width * center_x_ratio)
    center_y = int(height * center_y_ratio)

    # Distance au centre (normalisée)
    max_radius = min(height, width) * radius_ratio
    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)

    # Masque binaire initial (cercle)
    mask = (dist_from_center <= max_radius).astype(np.float32)

    # Ajouter des irrégularités avec du bruit de Perlin simulé
    # (approximation simple avec des sinus)
    noise = np.sin(x / 50.0) * np.cos(y / 30.0) * 0.1
    noise += np.sin(x / 30.0 + 2.5) * np.cos(y / 50.0 + 1.5) * 0.1

    # Appliquer le bruit pour déformer les bords
    dist_with_noise = dist_from_center + noise * max_radius
    mask = (dist_with_noise <= max_radius).astype(np.float32)

    # Lisser les bords pour une transition douce
    mask = gaussian_filter(mask, sigma=smoothness)

    # Normaliser entre 0 et 1
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

    return mask


def create_lighthouse_mask(height, width):
    """
    Crée un masque spécifique pour isoler le phare de son arrière-plan.
    Le phare est généralement au centre-gauche de l'image.

    Returns un masque où 1 = garder le phare (foreground), 0 = remplacer par la tempête
    """

    # Le phare est typiquement au centre-gauche
    # On crée un masque qui garde la région du phare
    mask = create_irregular_mask(
        height, width,
        center_x_ratio=0.35,  # Légèrement à gauche
        center_y_ratio=0.5,   # Centré verticalement
        radius_ratio=0.4,     # Rayon couvrant le phare
        smoothness=40         # Transition douce
    )

    return mask


# -------------------------------------------------
# VISUALISATION
# -------------------------------------------------

def display_blending_process(image1, image2, mask, blended,
                             l_stack1, l_stack2, blend_stack, sigmas,
                             save_path=None):
    """
    Affiche le processus complet de mélange multirésolution.

    4 rangées × (num_levels + 1) colonnes:
    - Rangée 0: Pile laplacienne image 1
    - Rangée 1: Pile laplacienne image 2
    - Rangée 2: Pile laplacienne mélangée
    - Rangée 3: [Images originales] + [Masque] + [Résultat]
    """

    num_levels = len(sigmas)

    fig, axes = plt.subplots(4, num_levels + 1, figsize=(3*(num_levels+1), 12))
    fig.suptitle('Processus de mélange multirésolution (Burt & Adelson, 1983)',
                 fontsize=14, fontweight='bold')

    # Rangées 0-2: Piles laplaciennes (num_levels colonnes)
    for level in range(num_levels):
        sigma_label = f'σ={sigmas[level]:.1f}'

        # Rangée 0: Pile laplacienne image 1
        l1 = l_stack1[level]
        if l1.ndim == 3:
            l1_display = (l1 - l1.min()) / (l1.max() - l1.min() + 1e-8)
        else:
            l1_display = (l1 - l1.min()) / (l1.max() - l1.min() + 1e-8)
        axes[0, level].imshow(np.clip(l1_display, 0, 1), cmap='gray' if l1.ndim == 2 else None)
        axes[0, level].set_title(sigma_label, fontsize=9)
        axes[0, level].axis('off')
        if level == 0:
            axes[0, level].set_ylabel('Laplacien\nImage 1', fontsize=9, rotation=0, ha='right', va='center')

        # Rangée 1: Pile laplacienne image 2
        l2 = l_stack2[level]
        if l2.ndim == 3:
            l2_display = (l2 - l2.min()) / (l2.max() - l2.min() + 1e-8)
        else:
            l2_display = (l2 - l2.min()) / (l2.max() - l2.min() + 1e-8)
        axes[1, level].imshow(np.clip(l2_display, 0, 1), cmap='gray' if l2.ndim == 2 else None)
        axes[1, level].axis('off')
        if level == 0:
            axes[1, level].set_ylabel('Laplacien\nImage 2', fontsize=9, rotation=0, ha='right', va='center')

        # Rangée 2: Pile laplacienne mélangée
        lb = blend_stack[level]
        if lb.ndim == 3:
            lb_display = (lb - lb.min()) / (lb.max() - lb.min() + 1e-8)
        else:
            lb_display = (lb - lb.min()) / (lb.max() - lb.min() + 1e-8)
        axes[2, level].imshow(np.clip(lb_display, 0, 1), cmap='gray' if lb.ndim == 2 else None)
        axes[2, level].axis('off')
        if level == 0:
            axes[2, level].set_ylabel('Laplacien\nMélangé', fontsize=9, rotation=0, ha='right', va='center')

    # Dernière colonne des rangées 0-2: vide
    for row in range(3):
        axes[row, -1].axis('off')

    # Rangée 3: Images originales + masque + résultat
    axes[3, 0].imshow(image1, cmap='gray' if image1.ndim == 2 else None)
    axes[3, 0].set_title('Image 1', fontsize=9)
    axes[3, 0].axis('off')

    axes[3, 1].imshow(image2, cmap='gray' if image2.ndim == 2 else None)
    axes[3, 1].set_title('Image 2', fontsize=9)
    axes[3, 1].axis('off')

    axes[3, 2].imshow(mask, cmap='gray')
    axes[3, 2].set_title('Masque', fontsize=9)
    axes[3, 2].axis('off')

    # Remplir les colonnes restantes avec le résultat
    for col in range(3, num_levels + 1):
        axes[3, col].imshow(blended, cmap='gray' if blended.ndim == 2 else None)
        if col == 3:
            axes[3, col].set_title('Résultat mélangé', fontsize=9)
        axes[3, col].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure sauvegardée : {save_path}")

    plt.show()


def display_comparison(image1, image2, blended_naive, blended_multi,
                      mask=None, save_path=None):
    """
    Compare le mélange naïf vs multirésolution.
    """

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Comparaison : Mélange naïf vs Multirésolution',
                 fontsize=14, fontweight='bold')

    # Rangée 0: Images sources et masque
    axes[0, 0].imshow(image1, cmap='gray' if image1.ndim == 2 else None)
    axes[0, 0].set_title('Image 1')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(image2, cmap='gray' if image2.ndim == 2 else None)
    axes[0, 1].set_title('Image 2')
    axes[0, 1].axis('off')

    if mask is not None:
        axes[0, 2].imshow(mask, cmap='gray')
        axes[0, 2].set_title('Masque')
        axes[0, 2].axis('off')
    else:
        axes[0, 2].axis('off')

    # Rangée 1: Comparaison résultats
    axes[1, 0].axis('off')

    axes[1, 1].imshow(blended_naive, cmap='gray' if blended_naive.ndim == 2 else None)
    axes[1, 1].set_title('Mélange naïf (bordure nette)')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(blended_multi, cmap='gray' if blended_multi.ndim == 2 else None)
    axes[1, 2].set_title('Mélange multirésolution (bordure lisse)')
    axes[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Figure sauvegardée : {save_path}")

    plt.show()


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    print("="*70)
    print("TP - MÉLANGE MULTIRÉSOLUTION (Burt & Adelson, 1983)")
    print("="*70)

    # ======================================================
    # PARTIE 3.1 : Mélange pomme + orange = "pommange"
    # ======================================================

    print("\n### PARTIE 3.1 : Mélange pomme + orange ###\n")

    # Charger les images
    try:
        apple = imageio.imread('./apple.jpeg')
        orange = imageio.imread('./orange.jpeg')
        print(f"Images chargées : Pomme {apple.shape}, Orange {orange.shape}")
    except FileNotFoundError:
        print("⚠️  Images apple.jpg ou orange.jpg introuvables.")
        print("    Veuillez placer ces images dans le répertoire courant.")
        return

    # Vérifier que les images ont la même taille
    if apple.shape != orange.shape:
        print(f"⚠️  Les images ont des tailles différentes.")
        print(f"    Pomme: {apple.shape}, Orange: {orange.shape}")
        print("    Redimensionnement de l'orange à la taille de la pomme...")
        from skimage.transform import resize
        orange = resize(orange, apple.shape, anti_aliasing=True)
        orange = (orange * 255).astype(np.uint8)

    # Créer un masque vertical (moitié gauche = pomme, moitié droite = orange)
    height, width = apple.shape[:2]
    mask = create_vertical_mask(height, width, split_ratio=0.5)

    print(f"Masque créé : {mask.shape}")
    print(f"  Pomme à gauche (1), Orange à droite (0)")

    # Mélange naïf (pour comparaison)
    print("\nCréation du mélange naïf...")
    apple_norm = apple.astype(np.float32) / 255.0
    orange_norm = orange.astype(np.float32) / 255.0
    mask_3d = mask[:, :, np.newaxis] if apple.ndim == 3 else mask
    blended_naive = apple_norm * mask_3d + orange_norm * (1 - mask_3d)

    # Mélange multirésolution
    print("\nCréation du mélange multirésolution...")
    num_levels = 6  # Utiliser 6 niveaux pour un bon lissage
    blended_multi, blend_stack, sigmas = multiresolution_blend(
        apple, orange, mask, num_levels=num_levels
    )

    # Sauvegarder le résultat
    result_path = os.path.join(output_dir, 'apple_orange_blend.png')
    imageio.imwrite(result_path, (blended_multi * 255).astype(np.uint8))
    print(f"\n✅ Image mélangée sauvegardée : {result_path}")

    # Visualiser le processus complet
    print("\nGénération de la visualisation du processus...")

    # Créer les piles laplaciennes pour la visualisation
    g_stack1, _ = gaussian_stack(apple, num_levels=num_levels)
    l_stack1 = laplacian_stack(g_stack1)
    g_stack2, _ = gaussian_stack(orange, num_levels=num_levels)
    l_stack2 = laplacian_stack(g_stack2)

    display_blending_process(
        apple_norm, orange_norm, mask, blended_multi,
        l_stack1, l_stack2, blend_stack, sigmas,
        save_path=os.path.join(output_dir, 'blending_process.png')
    )

    # Comparaison naïf vs multirésolution
    print("\nGénération de la comparaison naïf vs multirésolution...")
    display_comparison(
        apple_norm, orange_norm, blended_naive, blended_multi,
        mask=mask,
        save_path=os.path.join(output_dir, 'blending_comparison.png')
    )

    print("\n" + "="*70)
    print("FIN DU TRAITEMENT")
    print("="*70)
    print(f"\nFichiers générés dans {output_dir}/:")
    print("  - apple_orange_blend.png          : Image 'pommange' finale")
    print("  - blending_process.png            : Visualisation du processus complet")
    print("  - blending_comparison.png         : Comparaison naïf vs multirésolution")

    # ======================================================
    # PARTIE 3.2 : Mélange personnalisé (Phare + Océan calme)
    # ======================================================

    print("\n" + "="*70)
    print("### PARTIE 3.2 : Mélange personnalisé - Phare avec Océan Calme ###")
    print("="*70 + "\n")

    # Charger les images personnalisées
    try:
        lighthouse = imageio.imread('./Lighthouse.jpg')
        ocean = imageio.imread('./Ocean.jpg')
        print(f"Images chargées : Phare {lighthouse.shape}, Océan {ocean.shape}")
    except FileNotFoundError:
        print("⚠️  Images Lighthouse.png ou Ocean.jpg introuvables.")
        print("    Tentative avec les images dans /mnt/user-data/outputs/...")
        try:
            lighthouse = imageio.imread('/mnt/user-data/outputs/Lighthouse.png')
            ocean = imageio.imread('/mnt/user-data/outputs/Ocean.jpg')
            print(f"Images chargées : Phare {lighthouse.shape}, Océan {ocean.shape}")
        except:
            print("    Images introuvables. Section ignorée.")
            return

    # Redimensionner si nécessaire
    if lighthouse.shape != ocean.shape:
        print(f"⚠️  Redimensionnement nécessaire.")
        print(f"    Phare: {lighthouse.shape}, Océan: {ocean.shape}")
        from skimage.transform import resize
        target_height = min(lighthouse.shape[0], ocean.shape[0])
        target_width = min(lighthouse.shape[1], ocean.shape[1])

        lighthouse = resize(lighthouse, (target_height, target_width, 3), anti_aliasing=True)
        lighthouse = (lighthouse * 255).astype(np.uint8)
        ocean = resize(ocean, (target_height, target_width, 3), anti_aliasing=True)
        ocean = (ocean * 255).astype(np.uint8)
        print(f"    Nouvelles dimensions: {lighthouse.shape}")

    # Créer un masque irrégulier pour isoler le phare
    height, width = lighthouse.shape[:2]
    mask_custom = create_lighthouse_mask(height, width)

    print(f"Masque irrégulier créé : {mask_custom.shape}")
    print(f"  Masque centré pour capturer le phare")

    # Mélange multirésolution avec masque irrégulier
    print("\nMélange multirésolution avec masque irrégulier...")
    num_levels_custom = 6
    blended_custom, blend_stack_custom, sigmas_custom = multiresolution_blend(
        lighthouse, ocean, mask_custom, num_levels=num_levels_custom
    )

    # Sauvegarder le résultat
    result_custom_path = os.path.join(output_dir, 'lighthouse_ocean_blend.png')
    imageio.imwrite(result_custom_path, (blended_custom * 255).astype(np.uint8))
    print(f"\n✅ Image mélangée sauvegardée : {result_custom_path}")

    # Visualiser le processus complet
    print("\nGénération de la visualisation du processus (Figure 10)...")

    lighthouse_norm = lighthouse.astype(np.float32) / 255.0
    ocean_norm = ocean.astype(np.float32) / 255.0

    g_stack_lh, _ = gaussian_stack(lighthouse, num_levels=num_levels_custom)
    l_stack_lh = laplacian_stack(g_stack_lh)
    g_stack_oc, _ = gaussian_stack(ocean, num_levels=num_levels_custom)
    l_stack_oc = laplacian_stack(g_stack_oc)

    display_blending_process(
        lighthouse_norm, ocean_norm, mask_custom, blended_custom,
        l_stack_lh, l_stack_oc, blend_stack_custom, sigmas_custom,
        save_path=os.path.join(output_dir, 'lighthouse_ocean_process.png')
    )

    # Affichage comparatif
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Mélange personnalisé : Phare avec Océan Calme',
                 fontsize=14, fontweight='bold')

    axes[0, 0].imshow(lighthouse_norm)
    axes[0, 0].set_title('Image 1 — Phare (foreground)')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(ocean_norm)
    axes[0, 1].set_title('Image 2 — Océan calme (background)')
    axes[0, 1].axis('off')

    axes[1, 0].imshow(mask_custom, cmap='gray')
    axes[1, 0].set_title('Masque irrégulier\n(1 = phare, 0 = océan)')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(blended_custom)
    axes[1, 1].set_title('Résultat — Phare sur océan calme')
    axes[1, 1].axis('off')

    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'lighthouse_ocean_comparison.png')
    fig.savefig(comparison_path, dpi=150, bbox_inches='tight')
    print(f"✅ Comparaison sauvegardée : {comparison_path}")
    plt.show()

    print("\n" + "="*70)
    print("TRAITEMENT COMPLET TERMINÉ")
    print("="*70)
    print(f"\nFichiers générés pour le mélange personnalisé :")
    print(f"  - lighthouse_ocean_blend.png      : Phare sur océan calme")
    print(f"  - lighthouse_ocean_process.png    : Processus complet (Figure 10)")
    print(f"  - lighthouse_ocean_comparison.png : Images + masque + résultat")


if __name__ == "__main__":
    main()
