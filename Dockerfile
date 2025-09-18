# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV, MTCNN, and face recognition
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/kaxten-2090b-firebase-adminsdk-fbsvc-051943d1e1.json

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser -m
RUN chown -R appuser:appuser /app
RUN mkdir -p /home/appuser/.keras && chown -R appuser:appuser /home/appuser
USER appuser

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application (removed --reload for production)
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
