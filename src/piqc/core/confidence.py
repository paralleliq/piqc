"""
Model identification confidence scoring.

Provides evidence-based scoring for model identification
with detailed tracking of identification sources.
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from kubernetes.client import V1Volume

from piqc.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class IdentificationEvidence:
    """
    Evidence collected during model identification.

    Tracks all sources of model information and provides
    a confidence score based on the quality of evidence.
    """

    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    primary_method: str = "none"
    model_name: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


class ConfidenceScorer:
    """
    Scores model identification confidence based on evidence.

    Confidence levels:
    - 0.95-1.00: Explicit MODEL_NAME or --model env var
    - 0.85-0.94: CLI --model argument
    - 0.70-0.84: Volume path parsing
    - 0.50-0.69: Image tag inference
    - 0.00-0.49: Unknown/guessing
    """

    # Model path patterns for volume-based detection
    MODEL_PATH_PATTERNS = [
        re.compile(r'/models/([^/]+)', re.IGNORECASE),
        re.compile(r'/hub/models--([^/]+)--([^/]+)', re.IGNORECASE),
        re.compile(r'/([^/]*llama[^/]*)', re.IGNORECASE),
        re.compile(r'/([^/]*mistral[^/]*)', re.IGNORECASE),
        re.compile(r'/([^/]*qwen[^/]*)', re.IGNORECASE),
    ]

    # Image tag patterns for model inference
    IMAGE_MODEL_PATTERNS = [
        re.compile(r'llama-?(\d+)b', re.IGNORECASE),
        re.compile(r'mistral-?(\d+)b', re.IGNORECASE),
        re.compile(r'qwen-?(\d+)b', re.IGNORECASE),
    ]

    def score(
        self,
        env_vars: dict[str, str],
        cli_args: list[str],
        image: str,
        volumes: Optional[list[V1Volume]] = None,
    ) -> IdentificationEvidence:
        """
        Score model identification confidence and collect evidence.

        Args:
            env_vars: Environment variables from container.
            cli_args: Command line arguments from container.
            image: Container image name.
            volumes: Optional list of K8s volumes.

        Returns:
            IdentificationEvidence with confidence score and sources.
        """
        evidence = IdentificationEvidence()

        # Try methods in order of confidence (highest first)

        # 1. Explicit MODEL_NAME environment variable (0.95-1.00)
        model_name = self._check_env_model(env_vars)
        if model_name:
            evidence.model_name = model_name
            evidence.confidence = 0.98
            evidence.primary_method = "environment_variable"
            evidence.sources.append(f"env:MODEL_NAME={model_name}")
            return evidence

        # 2. CLI --model argument (0.85-0.94)
        model_name = self._check_cli_model(cli_args)
        if model_name:
            evidence.model_name = model_name
            evidence.confidence = 0.90
            evidence.primary_method = "cli_argument"
            evidence.sources.append(f"cli:--model={model_name}")
            return evidence

        # 3. Volume path parsing (0.70-0.84)
        model_name = self._check_volume_paths(volumes)
        if model_name:
            evidence.model_name = model_name
            evidence.confidence = 0.75
            evidence.primary_method = "volume_path"
            evidence.sources.append(f"volume:/models/{model_name}")
            return evidence

        # 4. Image tag inference (0.50-0.69)
        model_name = self._check_image_tag(image)
        if model_name:
            evidence.model_name = model_name
            evidence.confidence = 0.55
            evidence.primary_method = "image_inference"
            evidence.sources.append(f"image:{image}")
            evidence.warnings.append("Model name inferred from image tag, may be inaccurate")
            return evidence

        # 5. No identification possible
        evidence.confidence = 0.0
        evidence.primary_method = "none"
        evidence.warnings.append("Could not identify model from any source")

        return evidence

    def _check_env_model(self, env_vars: dict[str, str]) -> Optional[str]:
        """Check for model name in environment variables."""
        # Priority order for env var names
        model_env_vars = [
            "MODEL_NAME",
            "VLLM_MODEL",
            "HF_MODEL_NAME",
            "MODEL_ID",
            "SERVED_MODEL_NAME",
        ]

        for var in model_env_vars:
            if var in env_vars and env_vars[var]:
                return env_vars[var]

        return None

    def _check_cli_model(self, cli_args: list[str]) -> Optional[str]:
        """Extract model name from CLI arguments."""
        i = 0
        while i < len(cli_args):
            arg = cli_args[i]

            # Check --model value
            if arg == "--model" and i + 1 < len(cli_args):
                return cli_args[i + 1]

            # Check --model=value format
            if arg.startswith("--model="):
                return arg.split("=", 1)[1]

            i += 1

        return None

    def _check_volume_paths(
        self,
        volumes: Optional[list[V1Volume]],
    ) -> Optional[str]:
        """Infer model name from volume mount paths."""
        if not volumes:
            return None

        for volume in volumes:
            # Check PVC names
            if volume.persistent_volume_claim:
                pvc_name = volume.persistent_volume_claim.claim_name
                if pvc_name:
                    for pattern in self.MODEL_PATH_PATTERNS:
                        match = pattern.search(pvc_name)
                        if match:
                            return match.group(1)

            # Check config map names
            if volume.config_map:
                cm_name = volume.config_map.name
                if cm_name:
                    for pattern in self.MODEL_PATH_PATTERNS:
                        match = pattern.search(cm_name)
                        if match:
                            return match.group(1)

        return None

    def _check_image_tag(self, image: str) -> Optional[str]:
        """Infer model name from container image tag."""
        if not image:
            return None

        for pattern in self.IMAGE_MODEL_PATTERNS:
            match = pattern.search(image)
            if match:
                # Reconstruct model name from pattern match
                return match.group(0)

        return None


def enhance_model_confidence(
    existing_confidence: float,
    evidence: IdentificationEvidence,
) -> float:
    """
    Combine existing confidence with new evidence.

    Args:
        existing_confidence: Current confidence score.
        evidence: New identification evidence.

    Returns:
        Updated confidence score.
    """
    if evidence.confidence > existing_confidence:
        return evidence.confidence
    return existing_confidence
