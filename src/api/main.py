"""FastAPI application entry point with CORS and OpenAPI documentation."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router
from src.database.connection import init_db
from src.database.initialization import seed_database
from src.database.connection import SessionLocal
from src.utils.config import settings
from src.utils.logger import logger

app = FastAPI(
    title="Pricing Optimization System API",
    description=(
        "REST API for dynamic pricing recommendations, econometric price elasticity estimation, "
        "constrained profit/revenue optimization, and stochastic Monte Carlo simulation under demand uncertainty."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(router)  # Direct access also


@app.on_event("startup")
def on_startup():
    """Initializes and seeds database on API startup."""
    logger.info("Initializing Pricing Optimization System API...")
    init_db()
    db = SessionLocal()
    try:
        seed_database(db)
    finally:
        db.close()
    logger.info("Database initialized.")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
