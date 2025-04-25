import cv2
import numpy as np

def bytes_to_opencv(image_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    return cv2.resize(image, size)