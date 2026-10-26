# Multi-stage production Dockerfile for Pricing Optimization System
FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT_STREAMLIT=8501 \
    PORT_API=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY . .

# Seed initial database and sample data
RUN python -c "from src.data.loader import DataLoader; dl = DataLoader(); from src.database.initialization import seed_database; from src.database.connection import SessionLocal; db = SessionLocal(); seed_database(db); db.close()"

EXPOSE 8501 8000

# Default command launches Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
