

from ultralytics import YOLO
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import PIL.Image  as Image
import io
import base64
import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_PATH= os.join(BASE_DIR,"models",'Yolo.pt')
MOBILENET_PATH= os.join(BASE_DIR,"models",'MobileNetV3_rust_classifier.keras')

YOLO_MODEL=YOLO(YOLO_PATH)
MOBILENET_MODEL=load_model(MOBILENET_PATH)



def predict(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).resize((224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    proba = MOBILENET_MODEL.predict(x)[0]
    prediction = np.argmax(proba)
    return prediction,proba[prediction]
    

def detect(image_bytes):
    image  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    result = YOLO_MODEL(image)[0]
    img_result= Image.fromarray(result.plot()) 
    buffer=io.BytesIO()
    img_result.save(buffer,format="PNG")
    base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    uri =f"data:image/png;base64,{base64_str}"
    return uri
    






