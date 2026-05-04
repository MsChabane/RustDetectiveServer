from app import app
from fastapi.responses import JSONResponse
from fastapi import UploadFile,File
from app.process import detect,predict

SUPPORTED_FILES =['png','jpeg','jpg']
CLASSES = {0:'no rust',1:'rust'}

@app.get("/")
def index():
    return JSONResponse({
        "app":"Rust detection",
        "version":"1.0.0",
        "models":[{
            "Mobilenet":{
                "size":"Large",
                "version":3,
                "accuracy":92.0
            },
            "Yolo":{
                "size":"Medium",
                "version":9,      
            },
            
        }]
    })

@app.post("predict",response_class=JSONResponse)
def upload(image:UploadFile=File(...)):
    extension = image.filename.split(".")[-1]
    if extension not in SUPPORTED_FILES:
        return JSONResponse(content={"error": "Unsupported file type"},status_code=400)
    
    image_bytes = image.file.read()
    try:
        class_number,probability =predict(image_bytes)
        uri =None
        if class_number ==1 :
            uri =detect(image_bytes)
        response ={
            "prediction":CLASSES[class_number],
            "probability":round(probability,4)*100,
            "uri":uri
        }
        return JSONResponse(content=response,status_code=200)
    except Exception as exp:
        print(f"[ERROR] {str(exp)}")
        return JSONResponse(content={
            "error":"internal Server error"
            },status_code=500) 
        
    






