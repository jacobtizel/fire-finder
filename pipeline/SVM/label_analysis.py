import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def analyze_dataset_imbalance():
    mask_dir = r"BoWFireDataset\dataset\gt"

    total_white_pixels = 0
    total_black_pixels = 0

    for filename in os.listdir(mask_dir):
        if filename.endswith(".png") and filename.startswith("fire"):
            img_path = os.path.join(mask_dir, filename)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            white_pixels = np.sum(img == 255)
            black_pixels = np.sum(img == 0)

            total_white_pixels += white_pixels
            total_black_pixels += black_pixels

    total_pixels = total_white_pixels + total_black_pixels

    plt.figure(figsize=(6, 6))
    plt.pie(
        [total_white_pixels, total_black_pixels],
        labels=["Fire", "Non-Fire"],
        autopct="%1.1f%%",
        colors=["orange", "gray"],
    )
    plt.title("Fire vs Non-Fire Pixel Distribution")
    plt.show()


if __name__ == "__main__":
    analyze_dataset_imbalance()
