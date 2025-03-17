from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.router import router

app = FastAPI(
    title="Content Recommender Engine (CoRE)",
    version="1.0.0",
    docs_url="/", 
    redoc_url=None
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
async def root():
    return {"status": "success", "message": "Welcome to the Content Recommender Engine (CoRE)", "Visit": "http://localhost:5000/core"} 
