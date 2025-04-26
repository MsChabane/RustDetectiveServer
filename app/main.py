from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from models.mobilenet import classify_image
from models.yolov9 import detect_corrosion
from schemas import UnifiedResponse
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


@app.post("/analyze", response_model=UnifiedResponse)
async def analyze_image(file: UploadFile = File(...)):
    try:
        if not file.content_type.endswith((".png", ".jpg", ".jpeg", ".PNG", ".JPEG", ".JPG")):
            raise HTTPException(status_code=400, detail="An image file is required.")

        image_bytes = await file.read()

        classification_result = classify_image(image_bytes)

        # yolo
        detection_result = None
        if classification_result["class_name"] == "corrosion" and classification_result["confidence"] > 0.5:
            detection_result = detect_corrosion(image_bytes)

        return {
            "classification": classification_result,
            "detection": detection_result
        }

    except Exception as e:
        logger.error(f"Error : {e}")
        raise HTTPException(status_code=500, detail="Internal server Error.")

@app.get("/health")
def health_check():
    return {"status": "OK", "message": "API works."}


