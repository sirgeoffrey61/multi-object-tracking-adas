# Inference-only image for edge deployment (Jetson Orin / x86 + NVIDIA GPU).
FROM python:3.11-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

RUN mkdir -p data models deployment/jetson_orin

# Mount trained weights at runtime, e.g.:
#   -v ./runs/train/kitti_yolov8n/weights/best.pt:/app/models/best.pt
# Or TensorRT engine:
#   -v ./deployment/jetson_orin/best.engine:/app/models/best.engine
ENV YOLO_MODEL_PATH=models/best.pt
ENV API_HOST=0.0.0.0
ENV API_PORT=8000

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
