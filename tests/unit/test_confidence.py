"""
Unit tests for confidence scoring module.

Tests model identification confidence and evidence tracking.
"""

import pytest
from unittest.mock import Mock

from piqc.core.confidence import (
    ConfidenceScorer,
    IdentificationEvidence,
    enhance_model_confidence,
)


class TestIdentificationEvidence:
    """Tests for IdentificationEvidence dataclass."""

    def test_default_values(self) -> None:
        """Test default values."""
        evidence = IdentificationEvidence()
        assert evidence.model_name is None
        assert evidence.confidence == 0.0
        assert evidence.sources == []
        assert evidence.warnings == []
        assert evidence.primary_method == "none"

    def test_with_values(self) -> None:
        """Test with custom values."""
        evidence = IdentificationEvidence(
            model_name="meta-llama/Llama-2-7b-hf",
            confidence=0.95,
            sources=["env:MODEL_NAME=meta-llama/Llama-2-7b-hf"],
            primary_method="environment_variable",
        )
        assert evidence.model_name == "meta-llama/Llama-2-7b-hf"
        assert evidence.confidence == 0.95
        assert evidence.primary_method == "environment_variable"

    def test_multiple_sources(self) -> None:
        """Test with multiple sources."""
        evidence = IdentificationEvidence(
            model_name="test-model",
            confidence=0.98,
            sources=["env:MODEL_NAME=test", "cli:--model=test"],
        )
        assert len(evidence.sources) == 2


class TestConfidenceScorer:
    """Tests for ConfidenceScorer class."""

    def test_scorer_initialization(self) -> None:
        """Test scorer can be created."""
        scorer = ConfidenceScorer()
        assert scorer is not None

    def test_score_env_var_high_confidence(self) -> None:
        """Test env var source gives high confidence."""
        scorer = ConfidenceScorer()

        evidence = scorer.score(
            env_vars={"MODEL_NAME": "meta-llama/Llama-2-7b-hf"},
            cli_args=[],
            image="",
        )

        assert evidence.model_name == "meta-llama/Llama-2-7b-hf"
        assert evidence.confidence >= 0.95
        assert evidence.primary_method == "environment_variable"

    def test_score_vllm_model_env(self) -> None:
        """Test VLLM_MODEL env var is recognized."""
        scorer = ConfidenceScorer()

        evidence = scorer.score(
            env_vars={"VLLM_MODEL": "mistralai/Mistral-7B-v0.1"},
            cli_args=[],
            image="",
        )

        assert evidence.model_name == "mistralai/Mistral-7B-v0.1"
        assert evidence.confidence >= 0.95

    def test_score_cli_arg_high_confidence(self) -> None:
        """Test CLI arg source gives high confidence."""
        scorer = ConfidenceScorer()

        evidence = scorer.score(
            env_vars={},
            cli_args=["python", "-m", "vllm.entrypoints.openai.api_server",
                     "--model", "mistralai/Mistral-7B-v0.1"],
            image="",
        )

        assert evidence.model_name == "mistralai/Mistral-7B-v0.1"
        assert evidence.confidence >= 0.85
        assert evidence.primary_method == "cli_argument"

    def test_score_cli_arg_equals_format(self) -> None:
        """Test CLI --model=value format."""
        scorer = ConfidenceScorer()

        evidence = scorer.score(
            env_vars={},
            cli_args=["--model=meta-llama/Llama-2-13b-hf"],
            image="",
        )

        assert evidence.model_name == "meta-llama/Llama-2-13b-hf"
        assert evidence.confidence >= 0.85

    def test_score_image_tag_lower_confidence(self) -> None:
        """Test image tag gives lower confidence."""
        scorer = ConfidenceScorer()

        evidence = scorer.score(
            env_vars={},
            cli_args=[],
            image="vllm/vllm-openai:llama-7b",
        )

        # Image inference has lower confidence
        assert evidence.confidence < 0.7
        assert evidence.primary_method == "image_inference"

    def test_no_model_no_confidence(self) -> None:
        """Test no model found returns zero confidence."""
        scorer = ConfidenceScorer()

        evidence = scorer.score(
            env_vars={},
            cli_args=[],
            image="",
        )
        assert evidence.model_name is None
        assert evidence.confidence == 0.0
        assert evidence.primary_method == "none"

    def test_env_var_priority_over_cli(self) -> None:
        """Test env var is checked before CLI args."""
        scorer = ConfidenceScorer()

        evidence = scorer.score(
            env_vars={"MODEL_NAME": "env-model"},
            cli_args=["--model", "cli-model"],
            image="image-model",
        )

        # Env var should win
        assert evidence.model_name == "env-model"
        assert evidence.primary_method == "environment_variable"

    def test_cli_priority_over_image(self) -> None:
        """Test CLI args checked before image."""
        scorer = ConfidenceScorer()

        evidence = scorer.score(
            env_vars={},
            cli_args=["--model", "cli-model"],
            image="llama-7b",
        )

        # CLI should win
        assert evidence.model_name == "cli-model"
        assert evidence.primary_method == "cli_argument"

    def test_different_model_env_vars(self) -> None:
        """Test different MODEL env var keys are recognized."""
        scorer = ConfidenceScorer()

        test_cases = [
            ("MODEL_NAME", "model-a"),
            ("VLLM_MODEL", "model-b"),
            ("HF_MODEL_NAME", "model-c"),
            ("MODEL_ID", "model-d"),
            ("SERVED_MODEL_NAME", "model-e"),
        ]

        for env_key, model_name in test_cases:
            evidence = scorer.score(
                env_vars={env_key: model_name},
                cli_args=[],
                image="",
            )
            assert evidence.model_name == model_name, f"Failed for {env_key}"


class TestEnhanceModelConfidence:
    """Tests for enhance_model_confidence function."""

    def test_higher_evidence_wins(self) -> None:
        """Test higher confidence evidence is used."""
        evidence = IdentificationEvidence(
            model_name="new-model",
            confidence=0.9,
        )

        result = enhance_model_confidence(0.5, evidence)
        assert result == 0.9

    def test_keep_existing_if_higher(self) -> None:
        """Test existing confidence kept if higher."""
        evidence = IdentificationEvidence(
            model_name="new-model",
            confidence=0.5,
        )

        result = enhance_model_confidence(0.9, evidence)
        assert result == 0.9


class TestScorerEdgeCases:
    """Edge case tests for confidence scorer."""

    def test_empty_model_name_ignored(self) -> None:
        """Test empty model name in env is ignored."""
        scorer = ConfidenceScorer()

        evidence = scorer.score(
            env_vars={"MODEL_NAME": ""},
            cli_args=[],
            image="",
        )

        assert evidence.model_name is None

    def test_whitespace_model_name(self) -> None:
        """Test whitespace-only model is kept as-is."""
        scorer = ConfidenceScorer()

        evidence = scorer.score(
            env_vars={"MODEL_NAME": "  model-with-spaces  "},
            cli_args=[],
            image="",
        )

        assert evidence.model_name == "  model-with-spaces  "

    def test_volume_detection_with_none(self) -> None:
        """Test volume detection handles None gracefully."""
        scorer = ConfidenceScorer()

        evidence = scorer.score(
            env_vars={},
            cli_args=[],
            image="",
            volumes=None,
        )

        assert evidence.model_name is None

    def test_warnings_on_no_identification(self) -> None:
        """Test warnings generated when no model found."""
        scorer = ConfidenceScorer()

        evidence = scorer.score(
            env_vars={},
            cli_args=[],
            image="generic-image",
        )

        assert len(evidence.warnings) > 0

    def test_warnings_on_image_inference(self) -> None:
        """Test warnings added for image-based inference."""
        scorer = ConfidenceScorer()

        evidence = scorer.score(
            env_vars={},
            cli_args=[],
            image="vllm:llama-7b",
        )

        if evidence.primary_method == "image_inference":
            assert len(evidence.warnings) > 0
