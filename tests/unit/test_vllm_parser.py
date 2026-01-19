"""
Unit tests for vLLM parser.

Tests configuration parsing, architecture inference, and
parameter extraction.
"""

import pytest

from piqc.collectors.vllm_collector import VLLMCollector, VLLMConfig
from piqc.parsers.vllm_parser import (
    VLLMParser,
    infer_model_architecture,
    infer_model_parameters,
)


class TestVLLMCollector:
    """Tests for VLLMCollector."""
    
    def test_parse_env_vars_basic(self) -> None:
        """Test parsing basic environment variables."""
        collector = VLLMCollector()
        
        env_vars = {
            "MODEL_NAME": "meta-llama/Llama-2-7b-hf",
            "TENSOR_PARALLEL_SIZE": "2",
            "DTYPE": "float16",
        }
        
        config = collector.collect(env_vars, [])
        
        assert config.model_name == "meta-llama/Llama-2-7b-hf"
        assert config.tensor_parallel_size == 2
        assert config.precision == "float16"
    
    def test_parse_cli_args_basic(self) -> None:
        """Test parsing CLI arguments."""
        collector = VLLMCollector()
        
        args = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", "mistralai/Mistral-7B-v0.1",
            "--tensor-parallel-size", "4",
            "--dtype", "bfloat16",
            "--max-model-len", "4096",
        ]
        
        config = collector.collect({}, args)
        
        assert config.model_name == "mistralai/Mistral-7B-v0.1"
        assert config.tensor_parallel_size == 4
        assert config.precision == "bfloat16"
        assert config.max_model_len == 4096
    
    def test_parse_cli_args_with_equals(self) -> None:
        """Test parsing CLI arguments with = syntax."""
        collector = VLLMCollector()
        
        args = [
            "--model=meta-llama/Llama-2-70b-chat-hf",
            "--tensor-parallel-size=8",
            "--gpu-memory-utilization=0.9",
        ]
        
        config = collector.collect({}, args)
        
        assert config.model_name == "meta-llama/Llama-2-70b-chat-hf"
        assert config.tensor_parallel_size == 8
        assert config.gpu_memory_utilization == 0.9
    
    def test_cli_overrides_env(self) -> None:
        """Test that CLI arguments override environment variables."""
        collector = VLLMCollector()
        
        env_vars = {
            "MODEL_NAME": "env-model",
            "TENSOR_PARALLEL_SIZE": "2",
        }
        
        args = [
            "--model", "cli-model",
        ]
        
        config = collector.collect(env_vars, args)
        
        assert config.model_name == "cli-model"
        assert config.tensor_parallel_size == 2  # From env
    
    def test_parse_boolean_flags(self) -> None:
        """Test parsing boolean flags."""
        collector = VLLMCollector()
        
        args = [
            "--model", "test-model",
            "--trust-remote-code",
            "--enforce-eager",
        ]
        
        config = collector.collect({}, args)
        
        assert config.trust_remote_code is True
        assert config.enforce_eager is True
    
    def test_parse_quantization(self) -> None:
        """Test parsing quantization settings."""
        collector = VLLMCollector()
        
        args = [
            "--model", "test-model",
            "--quantization", "awq",
        ]
        
        config = collector.collect({}, args)
        
        assert config.quantization == "awq"
    
    def test_gpu_memory_utilization_percentage(self) -> None:
        """Test that percentage values are normalized to 0-1 range."""
        collector = VLLMCollector()
        
        args = [
            "--gpu-memory-utilization", "90",
        ]
        
        config = collector.collect({}, args)
        
        assert config.gpu_memory_utilization == 0.9
    
    def test_empty_inputs(self) -> None:
        """Test handling of empty inputs."""
        collector = VLLMCollector()
        
        config = collector.collect({}, [])
        
        assert config.model_name is None
        assert config.confidence == 0.0


class TestVLLMParser:
    """Tests for VLLMParser."""
    
    def test_parse_model_info(self) -> None:
        """Test parsing VLLMConfig to ModelInfo."""
        parser = VLLMParser()
        
        vllm_config = VLLMConfig(
            model_name="meta-llama/Llama-2-7b-chat-hf",
            served_model_name="llama-7b",
            confidence=0.8,
        )
        
        model_info = parser.parse_model_info(vllm_config)
        
        assert model_info.name == "meta-llama/Llama-2-7b-chat-hf"
        assert model_info.served_name == "llama-7b"
        assert model_info.source == "huggingface"
        assert model_info.architecture == "llama"
        assert model_info.parameters == "7B"
    
    def test_parse_inference_config(self) -> None:
        """Test parsing VLLMConfig to InferenceConfig."""
        parser = VLLMParser()
        
        vllm_config = VLLMConfig(
            precision="fp16",
            quantization="gptq",
            max_model_len=4096,
            tensor_parallel_size=4,
        )
        
        config = parser.parse_inference_config(vllm_config)
        
        assert config.precision == "float16"
        assert config.quantization == "gptq"
        assert config.max_model_len == 4096
        assert config.tensor_parallel_size == 4
    
    def test_normalize_precision(self) -> None:
        """Test precision normalization."""
        parser = VLLMParser()
        
        assert parser._normalize_precision("fp16") == "float16"
        assert parser._normalize_precision("bf16") == "bfloat16"
        assert parser._normalize_precision("float32") == "float32"
        assert parser._normalize_precision("half") == "float16"
        assert parser._normalize_precision("auto") == "auto"


class TestArchitectureInference:
    """Tests for model architecture inference."""
    
    @pytest.mark.parametrize("model_name,expected", [
        ("meta-llama/Llama-2-7b-hf", "llama"),
        ("codellama/CodeLlama-34b-Instruct-hf", "llama"),
        ("mistralai/Mistral-7B-v0.1", "mistral"),
        ("mistralai/Mixtral-8x7B-v0.1", "mixtral"),
        ("tiiuae/falcon-40b", "falcon"),
        ("bigscience/bloom-560m", "bloom"),
        ("mosaicml/mpt-7b", "mpt"),
        ("microsoft/phi-2", "phi"),
        ("google/gemma-7b", "gemma"),
        ("Qwen/Qwen-7B", "qwen"),
        ("unknown-model", None),
    ])
    def test_infer_architecture(self, model_name: str, expected: str | None) -> None:
        """Test architecture inference for various models."""
        result = infer_model_architecture(model_name)
        assert result == expected


class TestParameterInference:
    """Tests for model parameter count inference."""
    
    @pytest.mark.parametrize("model_name,expected", [
        ("meta-llama/Llama-2-7b-hf", "7B"),
        ("meta-llama/Llama-2-13b-chat-hf", "13B"),
        ("meta-llama/Llama-2-70b-hf", "70B"),
        ("bigscience/bloom-560m", "560M"),
        ("microsoft/phi-1.5b", "1.5B"),
        ("unknown-model", None),
    ])
    def test_infer_parameters(self, model_name: str, expected: str | None) -> None:
        """Test parameter count inference for various models."""
        result = infer_model_parameters(model_name)
        assert result == expected
