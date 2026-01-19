# Mock vLLM Inference Container
#
# This container simulates a vLLM inference server for integration testing.
# It provides:
# - Fake nvidia-smi output for GPU metrics testing
# - OpenAI-compatible API endpoint
# - vLLM-like environment variables

FROM python:3.11-slim

LABEL maintainer="ModelSpec Introspector"
LABEL description="Mock vLLM container for integration testing"

# Install minimal dependencies
RUN pip install --no-cache-dir flask

# Create fake nvidia-smi script
RUN mkdir -p /usr/local/bin
COPY fake-nvidia-smi.sh /usr/local/bin/nvidia-smi
RUN chmod +x /usr/local/bin/nvidia-smi

# Create mock vLLM server
COPY mock-vllm-server.py /app/server.py

WORKDIR /app

# Default vLLM environment variables
ENV VLLM_VERSION="0.2.7"
ENV MODEL_NAME="meta-llama/Llama-2-7b-hf"
ENV SERVED_MODEL_NAME="llama-7b"
ENV TENSOR_PARALLEL_SIZE="1"
ENV DTYPE="bfloat16"
ENV MAX_MODEL_LEN="4096"
ENV GPU_MEMORY_UTILIZATION="0.9"

EXPOSE 8000

# Entrypoint mimics vLLM startup
ENTRYPOINT ["python", "-m", "vllm.entrypoints.openai.api_server"]
CMD ["--model", "meta-llama/Llama-2-7b-hf", "--served-model-name", "llama-7b", "--port", "8000"]

# Note: The actual entrypoint runs the mock server (overridden at runtime)
