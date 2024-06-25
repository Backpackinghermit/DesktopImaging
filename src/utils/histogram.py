import cv2
import numpy as np

def calculate_histogram(image_path):
    """Calculates the histogram for an image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        np.array: The histogram data (counts for each pixel value).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Assuming grayscale for simplicity
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist
