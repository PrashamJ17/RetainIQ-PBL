"""
RetainIQ — FastAPI Application Entry Point
============================================
Starts the backend server and mounts all API routers.

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .ingestion import router as ingestion_router
from .api import router as api_router

app = FastAPI(
    title="RetainIQ API",
    description="Customer Retention Engine — Backend API",
    version="2.0.0",
)

# Allow the Next.js frontend (localhost:3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(ingestion_router)
app.include_router(api_router)


@app.get("/")
def root():
    return {"service": "RetainIQ API", "version": "2.0.0", "status": "running"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}
