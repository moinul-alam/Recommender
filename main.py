from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.router import router

app = FastAPI(
    title="Content Recommender Engine (CoRE)",
    version="1.0.0",
    docs_url="/", 
    redoc_url=None
)

# Enable CORS for all origins (temporary setup)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API router
app.include_router(router)

# Root endpoint for testing
@app.get("/")
async def root():
    return {"status": "success", "message": "Welcome to the Content Recommender Engine (CoRE)", "Visit": "http://localhost:5000/core"} 
