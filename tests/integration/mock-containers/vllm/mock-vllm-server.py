#!/usr/bin/env python3
"""
Mock vLLM OpenAI-compatible API server.

Provides minimal endpoints for health checks and model info.
"""

import json
import os
from flask import Flask, jsonify, request

app = Flask(__name__)

# Configuration from environment
MODEL_NAME = os.environ.get("MODEL_NAME", "unknown-model")
SERVED_MODEL_NAME = os.environ.get("SERVED_MODEL_NAME", MODEL_NAME)
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


@app.route("/v1/models", methods=["GET"])
def list_models():
    """List available models (OpenAI API compatible)."""
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": SERVED_MODEL_NAME,
                "object": "model",
                "created": 1700000000,
                "owned_by": "vllm",
                "root": MODEL_NAME,
                "parent": None,
                "max_model_len": MAX_MODEL_LEN,
                "permission": []
            }
        ]
    })


@app.route("/v1/completions", methods=["POST"])
def completions():
    """Completions endpoint (mock response)."""
    data = request.json or {}
    prompt = data.get("prompt", "")
    
    return jsonify({
        "id": "cmpl-test-12345",
        "object": "text_completion",
        "created": 1700000000,
        "model": SERVED_MODEL_NAME,
        "choices": [
            {
                "text": " This is a mock response from the test vLLM server.",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": 10,
            "total_tokens": len(prompt.split()) + 10
        }
    })


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    """Chat completions endpoint (mock response)."""
    return jsonify({
        "id": "chatcmpl-test-12345",
        "object": "chat.completion",
        "created": 1700000000,
        "model": SERVED_MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a mock response from the test vLLM server."
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20
        }
    })


@app.route("/metrics", methods=["GET"])
def metrics():
    """Prometheus metrics endpoint (mock)."""
    metrics_text = f"""
# HELP vllm_num_requests_running Number of requests running
# TYPE vllm_num_requests_running gauge
vllm_num_requests_running{{model_name="{SERVED_MODEL_NAME}"}} 5

# HELP vllm_num_requests_waiting Number of requests waiting
# TYPE vllm_num_requests_waiting gauge
vllm_num_requests_waiting{{model_name="{SERVED_MODEL_NAME}"}} 2

# HELP vllm_gpu_cache_usage_perc GPU KV-cache usage
# TYPE vllm_gpu_cache_usage_perc gauge
vllm_gpu_cache_usage_perc{{model_name="{SERVED_MODEL_NAME}"}} 0.75
"""
    return metrics_text, 200, {"Content-Type": "text/plain"}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting mock vLLM server on port {port}")
    print(f"Model: {MODEL_NAME}")
    print(f"Served as: {SERVED_MODEL_NAME}")
    print(f"Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
    app.run(host="0.0.0.0", port=port)
