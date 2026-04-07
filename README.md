# CloudNetGuard

AI-powered, explainable (XAI) and autonomous DNS security framework for cloud-native environments.

> **Phase 1 — Docker Compose Demo**  
> Anomaly detection + real-time XAI explanations + SDN response simulation across 5 microservices.

---

## Overview

CloudNetGuard operates under the **MAPE-K** loop (Monitor → Analyze → Plan → Execute → Knowledge) to autonomously detect and respond to DNS-based attacks.

**Detected threat types:**

| Attack | Mechanism | Detection signal |
|--------|-----------|-----------------|
| DNS Tunneling | Data exfiltration via DNS queries | High entropy, long query, TXT record type |
| DNS DDoS | Flood attack on DNS server | High response size, very low TTL |
| Domain Flux | DGA-generated random domains | High NXDOMAIN rate, low TTL, many unique domains |
| DNS Amplification | Small query → large response | Large response size ratio |

---

## Architecture

```
dns-collector → ai-engine → sdn-sim → dashboard
                                ↑
                    redis (message queue + cache)
```

| Service | Role |
|---------|------|
| **dns-collector** | Synthetic traffic generation + feature extraction |
| **ai-engine** | Autoencoder anomaly detection + SHAP/XAI explanations |
| **sdn-sim** | Rule-based + RL policy engine, SDN response simulation |
| **dashboard** | Real-time Plotly Dash monitoring UI |
| **redis** | Redis Streams inter-service messaging |

---

## Quick Start

**Requirements:** Docker Desktop with Compose v2

```bash
# 1. Clone
git clone https://github.com/<your-username>/CloudNetGuard.git
cd CloudNetGuard

# 2. Train the model (first time only)
cd scripts
python train_models.py --generate --n-samples 50000 --epochs-ae 50
cd ..

# 3. Start all services
docker compose up --build

# 4. Open dashboard
open http://localhost:8050
```

---

## Dashboard

Three pages accessible at `http://localhost:8050`:

- **Overview** — Live traffic chart, anomaly rate gauge, threat distribution, recent anomalies
- **Anomalies** — Time-series scatter, filterable detail table
- **XAI** — SHAP feature importance, waterfall chart, natural language explanation

SDN API available at `http://localhost:5002/actions`

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| ML | PyTorch 2.x |
| XAI | SHAP 0.44+ |
| DNS parsing | scapy 2.5+ |
| Message queue | Redis Streams 7.x |
| Web | Flask 3.x + Plotly Dash 2.x |
| Container | Docker Compose v2 |

---

## Training with Real Data

Supports [CIC-Bell-DNS-2021](https://www.unb.ca/cic/datasets/dns-2021.html) dataset:

```bash
# Download CSVs from UNB, then:
cd scripts
python train_models.py \
  --cic-dir /path/to/CSVs \
  --model autoencoder \
  --epochs-ae 60 \
  --max-benign 50000 \
  --max-anomaly 4000
```

---

## Project Roadmap

- [x] Phase 1 — Docker Compose demo (current)
- [ ] Phase 2 — Kubernetes operator (real cluster deployment)
- [ ] Phase 3 — eBPF/Cilium SDN integration
- [ ] Phase 4 — Federated learning across multiple clusters

---

## SDN API Reference

```bash
# Get last 50 actions
curl http://localhost:5002/actions

# Manual decision
curl -X POST http://localhost:5002/decide \
  -H "Content-Type: application/json" \
  -d '{"anomaly_score": 0.92, "predicted_type": "tunnel"}'

# Current policy
curl http://localhost:5002/policy
```

---

## License

MIT
