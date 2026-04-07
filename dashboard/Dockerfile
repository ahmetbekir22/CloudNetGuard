FROM python:3.11-slim AS builder
WORKDIR /build
COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY shared /app/shared
COPY dashboard/app.py ./
COPY dashboard/layouts/ ./layouts/
COPY dashboard/static/ ./static/
ENV PYTHONUNBUFFERED=1
EXPOSE 8050
CMD ["python", "app.py"]
