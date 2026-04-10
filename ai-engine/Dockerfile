FROM python:3.11-slim AS builder
WORKDIR /build
COPY ai-engine/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY shared /app/shared
COPY ai-engine/engine.py ai-engine/trainer.py ai-engine/explainer.py ./
COPY ai-engine/models/ ./models/
COPY ai-engine/pretrained/ ./pretrained/
ENV PYTHONUNBUFFERED=1
CMD ["python", "engine.py"]
