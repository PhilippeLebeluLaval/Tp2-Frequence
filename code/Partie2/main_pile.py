import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from scipy.ndimage import gaussian_filter
import os

def gaussian_stack(image, num_levels=5, sigma_base=2):

    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    else:
        image = image.astype(np.float32)

    # Déterminer la forme de la pile
    if image.ndim == 2:  # grayscale
        stack = np.zeros((num_levels, *image.shape), dtype=np.float32)
    else:  # RGB
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

    num_levels = gaussian_stack_array.shape[0]
    laplacian_stack_array = np.zeros_like(gaussian_stack_array)

    for level in range(num_levels - 1):
        laplacian_stack_array[level] = gaussian_stack_array[level] - gaussian_stack_array[level + 1]

    laplacian_stack_array[-1] = gaussian_stack_array[-1]

    return laplacian_stack_array


def display_stack(stack, sigmas, title, cmap='gray', save_path=None):
    num_levels = stack.shape[0]

    fig, axes = plt.subplots(1, num_levels, figsize=(3*num_levels, 3))
    if num_levels == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=14, fontweight='bold')

    for level in range(num_levels):
        ax = axes[level]

        img = stack[level]

        if cmap == 'gray' and img.ndim == 2:
            if img.min() < 0:
                img_display = img - img.min()
                img_display = img_display / (img_display.max() + 1e-8)
            else:
                img_display = img
            ax.imshow(img_display, cmap=cmap)
        else:
            ax.imshow(np.clip(img, 0, 1), cmap=cmap if img.ndim == 2 else None)

        ax.set_title(f'σ = {sigmas[level]:.1f}', fontsize=10)
        ax.axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure sauvegardée : {save_path}")

    plt.show()

def main():

    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("TP - PILES GAUSSIENNES ET LAPLACIENNES")
    print("="*60)
    print("\n\n### PARTIE 2.1 : Piles sur image multi-résolution ###\n")

    lincoln = imageio.imread('./lincoln.jpg', pilmode='L')
    print(f"Image Lincoln et Gala chargée : {lincoln.shape}")

    num_levels = 6
    g_stack_lincoln, sigmas_lincoln = gaussian_stack(lincoln, num_levels=num_levels, sigma_base=2)
    l_stack_lincoln = laplacian_stack(g_stack_lincoln)

    print(f"Pile gaussienne créée : {g_stack_lincoln.shape}")
    print(f"Sigmas utilisés : {sigmas_lincoln}")
    print(f"Pile laplacienne créée : {l_stack_lincoln.shape}")

    display_stack(
        g_stack_lincoln,
        sigmas_lincoln,
        'Pile Gaussienne — Lincoln et Gala (ou image de substitution)',
        cmap='gray',
        save_path=os.path.join(output_dir, 'lincoln_gaussian_stack.png')
    )

    display_stack(
        l_stack_lincoln,
        sigmas_lincoln,
        'Pile Laplacienne — Lincoln et Gala (ou image de substitution)',
        cmap='gray',
        save_path=os.path.join(output_dir, 'lincoln_laplacian_stack.png')
    )

    print("\n\n### PARTIE 2.2 : Piles sur l'image hybride ###\n")
    print("Utilisation de l'image hybride générée à la partie 1\n")


    hybrid_image = imageio.imread('Marilyn_Einstein_hybrid.png', pilmode='L')
    print(f"Image hybride chargée : {hybrid_image.shape}")


    # Créer les piles pour l'image hybride
    print("\n--- Création piles pour l'image hybride ---")
    g_stack_hybrid, sigmas_h = gaussian_stack(hybrid_image, num_levels=5, sigma_base=2)
    l_stack_hybrid = laplacian_stack(g_stack_hybrid)

    print(f"Pile gaussienne hybride créée : {g_stack_hybrid.shape}")
    print(f"Pile laplacienne hybride créée : {l_stack_hybrid.shape}")

    display_stack(
        g_stack_hybrid,
        sigmas_h,
        'Pile Gaussienne — Image Hybride (Marilyn/Einstein)',
        cmap='gray',
        save_path=os.path.join(output_dir, 'hybrid_gaussian_stack.png')
    )

    display_stack(
        l_stack_hybrid,
        sigmas_h,
        'Pile Laplacienne — Image Hybride (Marilyn/Einstein)',
        cmap='gray',
        save_path=os.path.join(output_dir, 'hybrid_laplacian_stack.png')
    )

    print("\n" + "="*60)
    print("FIN DU TRAITEMENT")
    print("="*60)


if __name__ == "__main__":
    main()