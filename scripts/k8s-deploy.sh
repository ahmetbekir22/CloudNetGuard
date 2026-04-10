#!/usr/bin/env bash
# CloudNetGuard — K8s deploy scripti (OrbStack / k3s)
# Kullanım: bash scripts/k8s-deploy.sh

set -e
cd "$(dirname "$0")/.."

IMAGES=(
  "localhost/cng-dns-collector:latest|dns-collector/Dockerfile"
  "localhost/cng-ai-engine:latest|ai-engine/Dockerfile"
  "localhost/cng-sdn-sim:latest|sdn-sim/Dockerfile"
  "localhost/cng-dashboard:latest|dashboard/Dockerfile"
)

echo "==> Docker imajları build ediliyor..."
for entry in "${IMAGES[@]}"; do
  IMAGE="${entry%%|*}"
  DOCKERFILE="${entry##*|}"
  echo "  Building $IMAGE"
  docker build -t "$IMAGE" -f "$DOCKERFILE" .
done

echo ""
echo "==> K8s manifests uygulanıyor..."
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmaps/
kubectl apply -f k8s/redis/
kubectl apply -f k8s/dns-collector/
kubectl apply -f k8s/ai-engine/
kubectl apply -f k8s/sdn-sim/
kubectl apply -f k8s/dashboard/

echo ""
echo "==> Pod durumu bekleniyor..."
kubectl rollout status deployment/dns-collector -n cloudnetguard --timeout=120s
kubectl rollout status deployment/ai-engine     -n cloudnetguard --timeout=120s
kubectl rollout status deployment/sdn-sim       -n cloudnetguard --timeout=120s
kubectl rollout status deployment/dashboard     -n cloudnetguard --timeout=120s

echo ""
echo "==> Çalışan pod'lar:"
kubectl get pods -n cloudnetguard

echo ""
echo "==> Servisler:"
kubectl get svc -n cloudnetguard

echo ""
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
echo "Dashboard: http://${NODE_IP}:30050"
echo "SDN API:   http://${NODE_IP}:30002"
