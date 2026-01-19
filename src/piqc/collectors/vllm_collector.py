"""
vLLM-specific configuration collection.

Extracts vLLM configuration from environment variables,
CLI arguments, and volume mounts.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from piqc.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class VLLMConfig:
    """
    vLLM configuration extracted from a deployment.
    
    Consolidates configuration from multiple sources:
    environment variables, CLI arguments, and volume mounts.
    """
    
    # Model Information
    model_name: Optional[str] = None
    served_model_name: Optional[str] = None
    model_path: Optional[str] = None
    tokenizer: Optional[str] = None
    trust_remote_code: bool = False
    
    # Inference Configuration
    precision: Optional[str] = None
    quantization: Optional[str] = None
    max_model_len: Optional[int] = None
    max_batch_tokens: Optional[int] = None
    max_sequences: Optional[int] = None
    
    # Parallelism
    tensor_parallel_size: Optional[int] = None
    pipeline_parallel_size: Optional[int] = None
    
    # Resource Management
    gpu_memory_utilization: Optional[float] = None
    swap_space_gb: Optional[int] = None
    
    # Engine Settings
    enforce_eager: bool = False
    kv_cache_dtype: Optional[str] = None
    
    # Detection Metadata
    detection_method: str = "unknown"
    confidence: float = 0.0
    collection_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    # Source tracking
    sources: dict[str, str] = field(default_factory=dict)


# Environment variable to config field mapping
ENV_VAR_MAPPING = {
    "MODEL_NAME": "model_name",
    "VLLM_MODEL": "model_name",
    "MODEL": "model_name",
    "SERVED_MODEL_NAME": "served_model_name",
    "TOKENIZER": "tokenizer",
    "TRUST_REMOTE_CODE": "trust_remote_code",
    "DOWNLOAD_DIR": "model_path",
    "DTYPE": "precision",
    "QUANTIZATION": "quantization",
    "MAX_MODEL_LEN": "max_model_len",
    "TENSOR_PARALLEL_SIZE": "tensor_parallel_size",
    "PIPELINE_PARALLEL_SIZE": "pipeline_parallel_size",
    "MAX_NUM_BATCHED_TOKENS": "max_batch_tokens",
    "MAX_NUM_SEQS": "max_sequences",
    "GPU_MEMORY_UTILIZATION": "gpu_memory_utilization",
    "SWAP_SPACE": "swap_space_gb",
    "KV_CACHE_DTYPE": "kv_cache_dtype",
}

# CLI argument to config field mapping
CLI_ARG_MAPPING = {
    "--model": "model_name",
    "--served-model-name": "served_model_name",
    "--tokenizer": "tokenizer",
    "--trust-remote-code": "trust_remote_code",
    "--download-dir": "model_path",
    "--dtype": "precision",
    "--quantization": "quantization",
    "--max-model-len": "max_model_len",
    "--tensor-parallel-size": "tensor_parallel_size",
    "--pipeline-parallel-size": "pipeline_parallel_size",
    "--max-num-batched-tokens": "max_batch_tokens",
    "--max-num-seqs": "max_sequences",
    "--gpu-memory-utilization": "gpu_memory_utilization",
    "--swap-space": "swap_space_gb",
    "--kv-cache-dtype": "kv_cache_dtype",
    "--enforce-eager": "enforce_eager",
}


class VLLMCollector:
    """
    Collects vLLM-specific configuration from deployments.
    
    Parses environment variables and CLI arguments to extract
    the complete vLLM configuration.
    """
    
    def collect(
        self,
        env_vars: dict[str, str],
        container_args: list[str],
    ) -> VLLMConfig:
        """
        Collect vLLM configuration from available sources.
        
        Args:
            env_vars: Environment variables from the container.
            container_args: Command line arguments from the container.
            
        Returns:
            VLLMConfig with extracted configuration.
        """
        config = VLLMConfig()
        sources: dict[str, str] = {}
        
        # Parse environment variables
        env_values = self._parse_env_vars(env_vars)
        for field_name, value in env_values.items():
            if value is not None:
                self._set_field(config, field_name, value)
                sources[field_name] = "env"
        
        # Parse CLI arguments (higher priority, can override env vars)
        cli_values = self._parse_cli_args(container_args)
        for field_name, value in cli_values.items():
            if value is not None:
                self._set_field(config, field_name, value)
                sources[field_name] = "cli"
        
        # Determine detection method
        config.sources = sources
        if sources:
            env_count = sum(1 for v in sources.values() if v == "env")
            cli_count = sum(1 for v in sources.values() if v == "cli")
            
            if env_count > 0 and cli_count > 0:
                config.detection_method = "mixed"
            elif cli_count > 0:
                config.detection_method = "cli_args"
            else:
                config.detection_method = "env_vars"
            
            # Calculate confidence based on critical fields
            config.confidence = self._calculate_confidence(config)
        
        # Infer model architecture if model name is known
        if config.model_name:
            config = self._infer_model_details(config)
        
        return config
    
    def _parse_env_vars(self, env_vars: dict[str, str]) -> dict[str, Any]:
        """Parse environment variables into config values."""
        values: dict[str, Any] = {}
        
        for env_name, field_name in ENV_VAR_MAPPING.items():
            if env_name in env_vars:
                raw_value = env_vars[env_name]
                parsed_value = self._parse_value(field_name, raw_value)
                if parsed_value is not None:
                    values[field_name] = parsed_value
        
        return values
    
    def _parse_cli_args(self, args: list[str]) -> dict[str, Any]:
        """Parse CLI arguments into config values."""
        values: dict[str, Any] = {}
        
        i = 0
        while i < len(args):
            arg = args[i]
            
            # Handle --flag=value format
            if "=" in arg:
                flag, value = arg.split("=", 1)
                if flag in CLI_ARG_MAPPING:
                    field_name = CLI_ARG_MAPPING[flag]
                    parsed_value = self._parse_value(field_name, value)
                    if parsed_value is not None:
                        values[field_name] = parsed_value
            
            # Handle --flag value format
            elif arg in CLI_ARG_MAPPING:
                field_name = CLI_ARG_MAPPING[arg]
                
                # Check if it's a boolean flag
                if field_name in ("trust_remote_code", "enforce_eager"):
                    values[field_name] = True
                elif i + 1 < len(args) and not args[i + 1].startswith("-"):
                    value = args[i + 1]
                    parsed_value = self._parse_value(field_name, value)
                    if parsed_value is not None:
                        values[field_name] = parsed_value
                    i += 1
            
            i += 1
        
        return values
    
    def _parse_value(self, field_name: str, raw_value: str) -> Any:
        """Parse a raw string value into the appropriate type."""
        raw_value = raw_value.strip()
        
        if not raw_value:
            return None
        
        # Integer fields
        if field_name in (
            "max_model_len",
            "max_batch_tokens",
            "max_sequences",
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "swap_space_gb",
        ):
            try:
                return int(raw_value)
            except ValueError:
                logger.debug(f"Failed to parse {field_name} as int: {raw_value}")
                return None
        
        # Float fields
        if field_name == "gpu_memory_utilization":
            try:
                value = float(raw_value)
                # Ensure value is between 0 and 1
                if value > 1.0:
                    value = value / 100.0
                return value
            except ValueError:
                logger.debug(f"Failed to parse {field_name} as float: {raw_value}")
                return None
        
        # Boolean fields
        if field_name in ("trust_remote_code", "enforce_eager"):
            return raw_value.lower() in ("true", "1", "yes")
        
        # String fields
        return raw_value
    
    def _set_field(self, config: VLLMConfig, field_name: str, value: Any) -> None:
        """Set a field on the config object."""
        if hasattr(config, field_name):
            setattr(config, field_name, value)
    
    def _calculate_confidence(self, config: VLLMConfig) -> float:
        """Calculate confidence score based on extracted configuration."""
        confidence = 0.0
        
        # Model name is critical
        if config.model_name:
            confidence += 0.4
        
        # Parallelism settings indicate vLLM
        if config.tensor_parallel_size or config.pipeline_parallel_size:
            confidence += 0.2
        
        # Precision/quantization settings
        if config.precision or config.quantization:
            confidence += 0.15
        
        # Memory settings
        if config.gpu_memory_utilization or config.max_model_len:
            confidence += 0.15
        
        # Served model name
        if config.served_model_name:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _infer_model_details(self, config: VLLMConfig) -> VLLMConfig:
        """
        Infer model architecture and parameters from model name.
        
        Note: Architecture and parameter inference is handled by the parser,
        not here. This method is kept for potential future use.
        """
        # Architecture and parameter inference is handled in vllm_parser.py
        # This method kept for potential future additions
        return config


def extract_model_source(model_name: Optional[str], model_path: Optional[str]) -> tuple[str, str]:
    """
    Determine model source from name or path.
    
    Args:
        model_name: Model name or identifier.
        model_path: Model path if available.
        
    Returns:
        Tuple of (source_type, source_path).
        source_type is one of: "huggingface", "s3", "gcs", "local", "unknown".
    """
    path_to_check = model_path or model_name or ""
    
    # Check for cloud storage
    if path_to_check.startswith("s3://"):
        return "s3", path_to_check
    
    if path_to_check.startswith("gs://"):
        return "gcs", path_to_check
    
    # Check for local path
    if path_to_check.startswith("/") or path_to_check.startswith("./"):
        return "local", path_to_check
    
    # Check for HuggingFace pattern (org/model or just model)
    if "/" in path_to_check or re.match(r"^[\w-]+$", path_to_check):
        return "huggingface", path_to_check
    
    return "unknown", path_to_check
