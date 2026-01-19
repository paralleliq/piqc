#!/bin/bash
#
# Setup script for ModelSpec Introspector integration test cluster
#
# Prerequisites:
#   - Docker installed and running
#   - kind installed (https://kind.sigs.k8s.io/)
#   - kubectl installed
#
# Usage:
#   ./setup-test-cluster.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
KIND_CONFIG="$SCRIPT_DIR/../kind/kind-config.yaml"
MANIFESTS_DIR="$SCRIPT_DIR/../kind/manifests"
CONTAINERS_DIR="$SCRIPT_DIR/../mock-containers"

CLUSTER_NAME="modelspec-test"

echo "==========================================="
echo "ModelSpec Introspector - Test Cluster Setup"
echo "==========================================="
echo ""

# Check prerequisites
check_prereqs() {
    echo "[1/6] Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        echo "ERROR: Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        echo "ERROR: Docker daemon is not running"
        exit 1
    fi
    
    if ! command -v kind &> /dev/null; then
        echo "ERROR: kind is not installed. Install from: https://kind.sigs.k8s.io/"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        echo "ERROR: kubectl is not installed"
        exit 1
    fi
    
    echo "       All prerequisites satisfied"
}

# Create kind cluster
create_cluster() {
    echo ""
    echo "[2/6] Creating kind cluster..."
    
    # Check if cluster exists
    if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        echo "       Cluster '$CLUSTER_NAME' already exists, recreating..."
        kind delete cluster --name "$CLUSTER_NAME"
    fi
    
    kind create cluster --config "$KIND_CONFIG"
    
    # Wait for cluster to be ready
    echo "       Waiting for cluster to be ready..."
    kubectl wait --for=condition=Ready nodes --all --timeout=120s
    
    echo "       Cluster created successfully"
}

# Build mock container images
build_images() {
    echo ""
    echo "[3/6] Building mock container images..."
    
    # vLLM mock
    echo "       Building vllm-mock..."
    docker build -t modelspec-test/vllm-mock:latest "$CONTAINERS_DIR/vllm/" -q
    
    # Triton mock
    echo "       Building triton-mock..."
    docker build -t modelspec-test/triton-mock:latest "$CONTAINERS_DIR/triton/" -q
    
    # TGI mock
    echo "       Building tgi-mock..."
    docker build -t modelspec-test/tgi-mock:latest "$CONTAINERS_DIR/tgi/" -q
    
    echo "       All images built"
}

# Load images into kind
load_images() {
    echo ""
    echo "[4/6] Loading images into kind cluster..."
    
    kind load docker-image modelspec-test/vllm-mock:latest --name "$CLUSTER_NAME"
    kind load docker-image modelspec-test/triton-mock:latest --name "$CLUSTER_NAME"
    kind load docker-image modelspec-test/tgi-mock:latest --name "$CLUSTER_NAME"
    
    echo "       All images loaded"
}

# Deploy test workloads
deploy_workloads() {
    echo ""
    echo "[5/6] Deploying test workloads..."
    
    # Apply all manifests
    kubectl apply -f "$MANIFESTS_DIR/vllm-llama-70b.yaml"
    kubectl apply -f "$MANIFESTS_DIR/vllm-mistral-7b.yaml"
    kubectl apply -f "$MANIFESTS_DIR/triton-server.yaml"
    kubectl apply -f "$MANIFESTS_DIR/tgi-falcon.yaml"
    kubectl apply -f "$MANIFESTS_DIR/generic-inference.yaml"
    kubectl apply -f "$MANIFESTS_DIR/nginx-control.yaml"
    
    echo "       All workloads deployed"
}

# Wait for pods to be ready
wait_for_pods() {
    echo ""
    echo "[6/6] Waiting for pods to be ready..."
    
    # Wait for inference namespace pods
    kubectl wait --for=condition=Ready pods --all -n inference --timeout=120s || true
    
    # Wait for web namespace pods
    kubectl wait --for=condition=Ready pods --all -n web --timeout=60s || true
    
    echo ""
    echo "==========================================="
    echo "Test cluster setup complete!"
    echo "==========================================="
    echo ""
    echo "Cluster: $CLUSTER_NAME"
    echo "Context: kind-$CLUSTER_NAME"
    echo ""
    echo "Deployed workloads:"
    kubectl get pods -A --selector='app' --no-headers | awk '{print "  - " $2 " (" $1 ")"}'
    echo ""
    echo "To run integration tests:"
    echo "  poetry run pytest tests/integration/ -v"
    echo ""
    echo "To tear down:"
    echo "  ./scripts/teardown-test-cluster.sh"
}

# Main execution
main() {
    check_prereqs
    create_cluster
    build_images
    load_images
    deploy_workloads
    wait_for_pods
}

main "$@"
