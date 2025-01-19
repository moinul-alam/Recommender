# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Content Recommender Engine (CoRE)",
        description="API for getting media recommendations for Explora",
        version="1.0.0"
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(api_router, prefix="/api")
    
    return app

app = create_app()
