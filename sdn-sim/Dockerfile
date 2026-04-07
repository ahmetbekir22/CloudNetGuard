FROM python:3.11-slim AS builder
WORKDIR /build
COPY sdn-sim/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY shared /app/shared
COPY sdn-sim/simulator.py sdn-sim/policy.py sdn-sim/actions.py ./
ENV PYTHONUNBUFFERED=1
EXPOSE 5001
CMD ["python", "simulator.py"]
