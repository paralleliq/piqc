#!/bin/bash
# Clean up fragmentation test resources
echo "Deleting test deployments..."
kubectl delete deployment mistral-7b llama-13b -n default --ignore-not-found
kubectl delete service mistral-7b -n default --ignore-not-found
kubectl delete job piqc-scan -n kube-system --ignore-not-found
echo "Done."
