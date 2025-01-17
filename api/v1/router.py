# api/v1/router.py
from fastapi import APIRouter
from api.v1.endpoints import content_based

api_router = APIRouter()
api_router.include_router(content_based.router, tags=["content-based"])
