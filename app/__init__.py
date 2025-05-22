from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI(title="Corrosion Detection API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


from App import routes