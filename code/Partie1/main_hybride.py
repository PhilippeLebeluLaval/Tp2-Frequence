import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os
from align_images import align_images
from hybrid_image import hybrid_image


def fft_magnitude(image):
    fft = np.fft.fft2(image)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.log(np.abs(fft_shift) + 1e-8)
    return magnitude


def process_pair(im_low, im_high, cutoff_low, cutoff_high, label_low, label_high):
    im_low, im_high = align_images(im_low, im_high)

    print(f"\n--- {label_low} / {label_high} ---")
    print(f"im_low  dtype: {im_low.dtype},  min: {im_low.min():.4f},  max: {im_low.max():.4f}")
    print(f"im_high dtype: {im_high.dtype}, min: {im_high.min():.4f}, max: {im_high.max():.4f}")

    hybrid = hybrid_image(im_low, im_high, cutoff_low, cutoff_high)

    im_low_f  = im_low.astype(np.float32)  / 255.0
    im_high_f = im_high.astype(np.float32) / 255.0
    hybrid_f  = hybrid.astype(np.float32)  / 255.0

    return im_low_f, im_high_f, hybrid_f


def save_pair(im_low_f, im_high_f, hybrid_f, label_low, label_high,
              cutoff_low, cutoff_high, output_dir='./output'):
    """
    Sauvegarde les 3 images (basse, haute, hybride) et la figure complète.
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = f"{label_low}_{label_high}"

    # Sauvegarde images individuelles
    imageio.imwrite(
        os.path.join(output_dir, f"{prefix}_low_{label_low}.png"),
        (im_low_f * 255).astype(np.uint8)
    )
    imageio.imwrite(
        os.path.join(output_dir, f"{prefix}_high_{label_high}.png"),
        (im_high_f * 255).astype(np.uint8)
    )
    imageio.imwrite(
        os.path.join(output_dir, f"{prefix}_hybrid.png"),
        (hybrid_f * 255).astype(np.uint8)
    )
    print(f"Images sauvegardées dans {output_dir}/")


def display_and_save_pair(im_low_f, im_high_f, hybrid_f,
                          label_low, label_high,
                          cutoff_low, cutoff_high,
                          output_dir='./output'):

    fft_low    = fft_magnitude(im_low_f)
    fft_high   = fft_magnitude(im_high_f)
    fft_hybrid = fft_magnitude(hybrid_f)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        f"Paire : {label_low} (bas) + {label_high} (haut) "
        f"— σ_low={cutoff_low}, σ_high={cutoff_high}",
        fontsize=13, fontweight='bold'
    )

    # --- Rangée 0 : images ---
    axes[0, 0].imshow(im_low_f,  cmap='gray')
    axes[0, 0].set_title(f"Basses fréquences\n({label_low})")

    axes[0, 1].imshow(im_high_f, cmap='gray')
    axes[0, 1].set_title(f"Hautes fréquences\n({label_high})")

    axes[0, 2].imshow(hybrid_f,  cmap='gray')
    axes[0, 2].set_title("Image Hybride\n(de près → haut, de loin → bas)")

    # --- Rangée 1 : FFTs ---
    axes[1, 0].imshow(fft_low,    cmap='inferno')
    axes[1, 0].set_title(f"FFT — {label_low}")

    axes[1, 1].imshow(fft_high,   cmap='inferno')
    axes[1, 1].set_title(f"FFT — {label_high}")

    axes[1, 2].imshow(fft_hybrid, cmap='inferno')
    axes[1, 2].set_title("FFT — Hybride")

    for ax in axes.ravel():
        ax.axis("off")

    plt.tight_layout()


    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, f"{label_low}_{label_high}_comparaison.png")
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Figure sauvegardée : {fig_path}")

    plt.show()
    plt.close(fig)


def main():

    output_dir = './output'

    im_marilyn  = imageio.imread('./Marilyn_Monroe.png', pilmode='L')
    im_einstein = imageio.imread('./Albert_Einstein.png', pilmode='L')

    cutoff_low_1  = 16
    cutoff_high_1 = 8

    low1, high1, hybrid1 = process_pair(
        im_marilyn, im_einstein,
        cutoff_low_1, cutoff_high_1,
        "Marilyn", "Einstein"
    )

    save_pair(low1, high1, hybrid1,
              "Marilyn", "Einstein",
              cutoff_low_1, cutoff_high_1,
              output_dir)

    display_and_save_pair(low1, high1, hybrid1,
                          "Marilyn", "Einstein",
                          cutoff_low_1, cutoff_high_1,
                          output_dir)


    im_lion = imageio.imread('./Lion.jpg',  pilmode='L')
    im_chat = imageio.imread('./Chat.jpg',  pilmode='L')

    cutoff_low_2  = 6
    cutoff_high_2 = 4

    low2, high2, hybrid2 = process_pair(
        im_lion, im_chat,
        cutoff_low_2, cutoff_high_2,
        "Lion", "Chat"
    )

    save_pair(low2, high2, hybrid2,
              "Lion", "Chat",
              cutoff_low_2, cutoff_high_2,
              output_dir)

    display_and_save_pair(low2, high2, hybrid2,
                          "Lion", "Chat",
                          cutoff_low_2, cutoff_high_2,
                          output_dir)


if __name__ == "__main__":
    main()