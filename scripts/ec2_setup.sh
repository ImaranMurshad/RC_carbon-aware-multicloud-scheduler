#!/usr/bin/env bash
# One-time setup of the thesis experiment host (Ubuntu on AWS EC2).
#
# Result: three Kubernetes clusters on this one machine, matching config/clusters.json
#   aws-k3s         the real k3s cluster already installed here
#   kind-azure-sim  Kind cluster standing in for Azure
#   kind-gcp-sim    Kind cluster standing in for GCP
# plus the project repo and a Python virtual environment.
#
# Safe to re-run (each step skips work that is already done).
# Run as the normal 'ubuntu' user, NOT with sudo:   bash ec2_setup.sh
set -euo pipefail

REPO_URL="https://github.com/ImaranMurshad/RC_carbon-aware-multicloud-scheduler.git"
PROJECT_DIR="$HOME/RC_carbon-aware-multicloud-scheduler"
export KUBECONFIG="$HOME/.kube/config"

if [ "$(id -u)" -eq 0 ]; then
  echo "Run this as the ubuntu user, not root / sudo." >&2
  exit 1
fi

echo "== 1/7 system packages"
sudo apt-get update -y
sudo apt-get install -y docker.io git python3-venv python3-pip curl jq

echo "== 2/7 docker"
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"
if ! docker ps >/dev/null 2>&1; then
  echo
  echo "Docker is installed, but this shell does not have the 'docker' group yet."
  echo "Log out and SSH in again, then re-run:  bash ec2_setup.sh"
  exit 0
fi

echo "== 3/7 kind"
if ! command -v kind >/dev/null 2>&1; then
  curl -fsSL -o /tmp/kind \
    https://github.com/kubernetes-sigs/kind/releases/latest/download/kind-linux-amd64
  sudo install -m 0755 /tmp/kind /usr/local/bin/kind
  rm -f /tmp/kind
fi
kind version

echo "== 4/7 kubeconfig entry for the k3s cluster (context aws-k3s)"
mkdir -p "$HOME/.kube"
touch "$KUBECONFIG"; chmod 600 "$KUBECONFIG"
if ! kubectl config get-contexts -o name 2>/dev/null | grep -qx "aws-k3s"; then
  sudo cat /etc/rancher/k3s/k3s.yaml | sed 's/: default/: aws-k3s/g' > /tmp/k3s-aws.yaml
  KUBECONFIG="$HOME/.kube/config:/tmp/k3s-aws.yaml" kubectl config view --flatten > /tmp/kube-merged
  install -m 600 /tmp/kube-merged "$HOME/.kube/config"
  rm -f /tmp/k3s-aws.yaml /tmp/kube-merged
fi

echo "== 5/7 Kind clusters (azure-sim, gcp-sim)"
for name in azure-sim gcp-sim; do
  if kind get clusters 2>/dev/null | grep -qx "$name"; then
    echo "cluster $name already exists"
  else
    kind create cluster --name "$name" --wait 180s
  fi
done

echo "== 6/7 metrics-server on the Kind clusters (needed for 'kubectl top')"
for ctx in kind-azure-sim kind-gcp-sim; do
  kubectl --context "$ctx" apply -f \
    https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
  if ! kubectl --context "$ctx" -n kube-system get deployment metrics-server \
        -o jsonpath='{.spec.template.spec.containers[0].args}' | grep -q "kubelet-insecure-tls"; then
    kubectl --context "$ctx" -n kube-system patch deployment metrics-server --type=json \
      -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
  fi
done

echo "== 7/7 project repo and Python environment"
if [ ! -d "$PROJECT_DIR/.git" ]; then
  git clone "$REPO_URL" "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"
git pull --ff-only
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt

echo
echo "================ CHECK ================"
kubectl config get-contexts
for c in aws-k3s kind-azure-sim kind-gcp-sim; do
  echo "--- $c"; kubectl --context "$c" get nodes
done
free -m
df -h /
echo
echo "Next: from your Mac, copy the .env file to the server (never via git):"
echo "  scp -i <key>.pem .env ubuntu@<ip>:$PROJECT_DIR/.env"