import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path


def save_image(path, image):
    cv2.imwrite(path, image)
    cv2.waitKey(10)


def bgr2gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def cv2pil(image_array: np.ndarray) -> Image:
    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_array)


def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    return new_image


def show_image_from_ndarray(image_array):
    if (len(image_array.shape) < 3) or (image_array.shape[-1] == 1):
        plt.imshow(image_array)
        plt.axis("off")
    else:
        plt.imshow(cv2pil(image_array))
        plt.axis("off")


def draw_multiple_image(titles, images):
    n_images = len(images)
    assert len(titles) == n_images

    fig, axes = plt.subplots(1, n_images)
    for i in range(n_images):
        axes[i].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        axes[i].set_title(titles[i])
        axes[i].axis("off")


def rotate_anticlockwise_90(image):
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def morphology_closing(mask, kernel_size=15):
    mask_processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), dtype=np.int16))
    return mask_processed


def morphology_dilatetion(mask, kernel_size=15):
    mask_processed = cv2.dilate(mask, np.ones((kernel_size, kernel_size), dtype=np.int16))
    return mask_processed


def gray2bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
