import cv2
import numpy as np
import torch

def load_model():
    return torch.load("app/models/weights/modelyolov9.pt")

def detect_corrosion(image_bytes: bytes) -> dict:
    model = load_model()

    # Convertir l'image en format OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img)

    bboxes = results.xyxy[0].numpy()  # Format: [xmin, ymin, xmax, ymax, confidence, class]

    corrosion_results = [box for box in bboxes if box[5] == 0]  #classe 0 = corrosion

    return {
        "bboxes": [box[:4].tolist() for box in corrosion_results],
        "scores": [box[4].tolist() for box in corrosion_results],
        "labels": ["corrosion"] * len(corrosion_results)
    }

