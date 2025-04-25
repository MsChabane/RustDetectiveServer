from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.mobilenet import classify_image as mobilenet_classify
from models.yolov9 import detect_corrosion as yolov9_detect
from schemas import ClassificationResponse, DetectionResponse
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Corrosion Detection API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/classify", response_model=ClassificationResponse)
async def classify(file: UploadFile = File(...)):
    try:
        if not file.content_type.endswith((".png", ".jpg", ".jpeg", ".PNG", ".JPEG", ".JPG")):
            raise HTTPException(status_code=400, detail="An image file is required.")

        image_bytes = await file.read()
        result = mobilenet_classify(image_bytes)
        return result
    except Exception as e:
        logger.error(f"Erreur classification: {e}")
        raise HTTPException(status_code=500, detail="Internal server Erreur.")


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)):
    try:
        if not file.content_type.endswith((".png", ".jpg", ".jpeg", ".PNG", ".JPEG", ".JPG")):
            raise HTTPException(status_code=400, detail="An image file is required.")

        image_bytes = await file.read()
        result = yolov9_detect(image_bytes)
        return result
    except Exception as e:
        logger.error(f"Detection erreur: {e}")
        raise HTTPException(status_code=500, detail="Internal server Erreur.")


@app.get("/health")
def health_check():
    return {"status": "OK", "message": "API works."}


