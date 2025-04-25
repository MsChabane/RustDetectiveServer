from pydantic import BaseModel

class ClassificationResponse(BaseModel):
    class_name: str
    confidence: float

class DetectionResponse(BaseModel):
    bboxes: list[list[float]]  # Format: [xmin, ymin, xmax, ymax]
    scores: list[float]
    labels: list[str]

