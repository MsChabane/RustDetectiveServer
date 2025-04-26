from pydantic import BaseModel

class ClassificationResult(BaseModel):
    class_name: str
    confidence: float

class DetectionResult(BaseModel):
    bboxes: list[list[float]]  # [xmin, ymin, xmax, ymax]
    scores: list[float]
    labels: list[str]

class UnifiedResponse(BaseModel):
    classification: ClassificationResult
    detection: DetectionResult