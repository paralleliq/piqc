#!/bin/bash
#
# Teardown script for ModelSpec Introspector integration test cluster
#
# Usage:
#   ./teardown-test-cluster.sh
#

set -e

CLUSTER_NAME="modelspec-test"

echo "==========================================="
echo "ModelSpec Introspector - Test Cluster Teardown"
echo "==========================================="
echo ""

# Delete kind cluster
echo "[1/2] Deleting kind cluster..."

if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
    kind delete cluster --name "$CLUSTER_NAME"
    echo "       Cluster deleted"
else
    echo "       Cluster '$CLUSTER_NAME' not found (already deleted?)"
fi

# Clean up docker images (optional)
echo ""
echo "[2/2] Cleaning up docker images..."

docker rmi modelspec-test/vllm-mock:latest 2>/dev/null || true
docker rmi modelspec-test/triton-mock:latest 2>/dev/null || true
docker rmi modelspec-test/tgi-mock:latest 2>/dev/null || true

echo "       Images cleaned"

echo ""
echo "==========================================="
echo "Teardown complete!"
echo "==========================================="
