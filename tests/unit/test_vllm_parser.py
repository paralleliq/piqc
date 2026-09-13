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

    def test_parse_enable_chunked_prefill(self) -> None:
        """--enable-chunked-prefill is a boolean flag, same shape as
        --enforce-eager above."""
        collector = VLLMCollector()

        config = collector.collect({}, ["--model", "test-model", "--enable-chunked-prefill"])

        assert config.enable_chunked_prefill is True

    def test_parse_kv_transfer_config_producer(self) -> None:
        """--kv-transfer-config is a JSON blob; kv_role/kv_connector are
        parsed out of it, not copied as a scalar."""
        collector = VLLMCollector()
        kv_config = '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'

        config = collector.collect({}, ["--model", "test-model", "--kv-transfer-config", kv_config])

        assert config.kv_role == "kv_producer"
        assert config.kv_connector == "PyNcclConnector"
        assert config.lmcache_enabled is None  # PyNcclConnector isn't LMCache

    def test_parse_kv_transfer_config_lmcache_connector(self) -> None:
        """lmcache_enabled is derived from the connector name inside the
        same --kv-transfer-config JSON, not a separate flag."""
        collector = VLLMCollector()
        kv_config = '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer"}'

        config = collector.collect({}, ["--model", "test-model", "--kv-transfer-config", kv_config])

        assert config.kv_role == "kv_consumer"
        assert config.lmcache_enabled is True

    def test_lmcache_enabled_from_env_var(self) -> None:
        """The other positive-evidence path for LMCache: an LMCACHE_* env
        var, independent of the KV connector name."""
        collector = VLLMCollector()

        config = collector.collect({"LMCACHE_CONFIG_FILE": "/etc/lmcache.yaml"}, ["--model", "test-model"])

        assert config.lmcache_enabled is True

    def test_malformed_kv_transfer_config_stays_unknown(self) -> None:
        """Malformed JSON must never be guessed at -- kv_role/kv_connector
        stay None, same as if the flag were absent entirely."""
        collector = VLLMCollector()

        config = collector.collect({}, ["--model", "test-model", "--kv-transfer-config", "{not valid json"])

        assert config.kv_role is None
        assert config.kv_connector is None

    def test_unrecognized_kv_role_value_stays_unknown(self) -> None:
        """A kv_role value outside vLLM's own three real values is treated
        as unparseable, not passed through -- see _VALID_KV_ROLES."""
        collector = VLLMCollector()
        kv_config = '{"kv_connector":"SomeConnector","kv_role":"not_a_real_role"}'

        config = collector.collect({}, ["--model", "test-model", "--kv-transfer-config", kv_config])

        assert config.kv_role is None


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

    def test_parse_inference_config_carries_kv_transfer_fields(self) -> None:
        """enable_chunked_prefill/kv_role/lmcache_enabled must survive the
        VLLMConfig -> InferenceConfig conversion, same as every other field
        this test class already checks."""
        parser = VLLMParser()

        vllm_config = VLLMConfig(
            enable_chunked_prefill=True,
            kv_role="kv_both",
            lmcache_enabled=True,
        )

        config = parser.parse_inference_config(vllm_config)

        assert config.enable_chunked_prefill is True
        assert config.kv_role == "kv_both"
        assert config.lmcache_enabled is True

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
