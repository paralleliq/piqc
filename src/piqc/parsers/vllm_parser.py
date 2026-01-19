"""
vLLM configuration parser.

Parses vLLM-specific configuration data into structured ModelSpec fields.
"""

import re
from typing import Optional

from piqc.collectors.vllm_collector import VLLMConfig, extract_model_source
from piqc.models.modelspec import ModelInfo, InferenceConfig
from piqc.utils.logger import get_logger


logger = get_logger(__name__)


# Model architecture patterns for inference
# NOTE: Order matters! More specific patterns must come first
ARCHITECTURE_PATTERNS = {
    "llama": [r"llama", r"codellama", r"code-llama"],
    "mixtral": [r"mixtral"],  # Must be before mistral
    "mistral": [r"mistral"],
    "falcon": [r"falcon"],
    "gpt-neox": [r"gpt-neox", r"pythia"],
    "gpt-j": [r"gpt-j"],
    "gpt2": [r"gpt2"],
    "opt": [r"\bopt-", r"\bopt\b"],
    "bloom": [r"bloom"],
    "mpt": [r"\bmpt-", r"\bmpt\b"],
    "phi": [r"\bphi-", r"\bphi\b"],
    "gemma": [r"gemma"],
    "qwen": [r"qwen"],
    "yi": [r"\byi-", r"\byi\b"],
    "vicuna": [r"vicuna"],
    "deepseek": [r"deepseek"],
    "starcoder": [r"starcoder"],
}

# Parameter count patterns
# NOTE: Decimal pattern must come before integer pattern
PARAM_PATTERNS = [
    (re.compile(r"(\d+\.\d+)b(?:[-_]|$)", re.IGNORECASE), lambda m: f"{m.group(1)}B"),
    (re.compile(r"(\d+)b(?:[-_]|$)", re.IGNORECASE), lambda m: f"{m.group(1)}B"),
    (re.compile(r"(\d+)m(?:[-_]|$)", re.IGNORECASE), lambda m: f"{m.group(1)}M"),
]


class VLLMParser:
    """
    Parses vLLM configuration into ModelSpec fields.
    
    Transforms collected vLLM configuration data into
    structured ModelInfo and InferenceConfig objects.
    """
    
    def parse_model_info(self, vllm_config: VLLMConfig) -> ModelInfo:
        """
        Parse vLLM config into ModelInfo.
        
        Args:
            vllm_config: Collected vLLM configuration.
            
        Returns:
            ModelInfo with parsed data.
        """
        model_name = vllm_config.model_name
        
        # Determine source
        source, source_path = extract_model_source(
            model_name,
            vllm_config.model_path,
        )
        
        # Infer architecture
        architecture = self._infer_architecture(model_name)
        
        # Infer parameter count
        parameters = self._infer_parameters(model_name)
        
        # Determine identification method
        if vllm_config.sources:
            methods = list(set(vllm_config.sources.values()))
            if len(methods) == 1:
                identification_method = methods[0]
            else:
                identification_method = "mixed"
        else:
            identification_method = "inferred"
        
        return ModelInfo(
            name=model_name,
            served_name=vllm_config.served_model_name,
            source=source,
            source_path=source_path,
            architecture=architecture,
            parameters=parameters,
            confidence=vllm_config.confidence,
            identification_method=identification_method,
        )
    
    def parse_inference_config(self, vllm_config: VLLMConfig) -> InferenceConfig:
        """
        Parse vLLM config into InferenceConfig.
        
        Args:
            vllm_config: Collected vLLM configuration.
            
        Returns:
            InferenceConfig with parsed data.
        """
        # Normalize precision
        precision = self._normalize_precision(vllm_config.precision)
        
        # Normalize quantization
        quantization = self._normalize_quantization(vllm_config.quantization)
        
        return InferenceConfig(
            precision=precision,
            quantization=quantization,
            max_model_len=vllm_config.max_model_len,
            max_batch_tokens=vllm_config.max_batch_tokens,
            max_sequences=vllm_config.max_sequences,
            tensor_parallel_size=vllm_config.tensor_parallel_size,
            pipeline_parallel_size=vllm_config.pipeline_parallel_size,
        )
    
    def _infer_architecture(self, model_name: Optional[str]) -> Optional[str]:
        """Infer model architecture from model name."""
        if not model_name:
            return None
        
        model_lower = model_name.lower()
        
        for arch, patterns in ARCHITECTURE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, model_lower):
                    return arch
        
        return None
    
    def _infer_parameters(self, model_name: Optional[str]) -> Optional[str]:
        """Infer parameter count from model name."""
        if not model_name:
            return None
        
        for pattern, formatter in PARAM_PATTERNS:
            match = pattern.search(model_name)
            if match:
                return formatter(match)
        
        return None
    
    def _normalize_precision(self, precision: Optional[str]) -> Optional[str]:
        """Normalize precision string to standard format."""
        if not precision:
            return None
        
        precision_lower = precision.lower().strip()
        
        # Map common variations to standard names
        precision_map = {
            "float16": "float16",
            "fp16": "float16",
            "half": "float16",
            "bfloat16": "bfloat16",
            "bf16": "bfloat16",
            "float32": "float32",
            "fp32": "float32",
            "full": "float32",
            "int8": "int8",
            "int4": "int4",
            "auto": "auto",
        }
        
        return precision_map.get(precision_lower, precision)
    
    def _normalize_quantization(self, quantization: Optional[str]) -> Optional[str]:
        """Normalize quantization method string."""
        if not quantization:
            return None
        
        quant_lower = quantization.lower().strip()
        
        # Map common variations
        quant_map = {
            "awq": "awq",
            "gptq": "gptq",
            "squeezellm": "squeezellm",
            "marlin": "marlin",
            "fp8": "fp8",
            "none": None,
        }
        
        return quant_map.get(quant_lower, quantization)


def infer_model_architecture(model_name: str) -> Optional[str]:
    """
    Infer model architecture from model name.
    
    Standalone function for use outside VLLMParser.
    
    Args:
        model_name: Model name or identifier.
        
    Returns:
        Architecture name or None if not recognized.
    """
    parser = VLLMParser()
    return parser._infer_architecture(model_name)


def infer_model_parameters(model_name: str) -> Optional[str]:
    """
    Infer parameter count from model name.
    
    Standalone function for use outside VLLMParser.
    
    Args:
        model_name: Model name or identifier.
        
    Returns:
        Parameter count string (e.g., "7B") or None.
    """
    parser = VLLMParser()
    return parser._infer_parameters(model_name)
