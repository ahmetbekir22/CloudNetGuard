FROM python:3.11-slim AS builder
WORKDIR /build
COPY dns-collector/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY shared /app/shared
COPY dns-collector/collector.py dns-collector/synthetic.py dns-collector/feature_extractor.py ./
RUN mkdir -p /app/data
ENV PYTHONUNBUFFERED=1
CMD ["python", "collector.py"]
