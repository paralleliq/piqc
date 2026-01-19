"""
Unit tests for execution modes module.

Tests execution mode detection, capabilities, and context.
"""

import os
import pytest
from unittest.mock import patch

from piqc.core.modes import (
    ExecutionMode,
    ModeCapabilities,
    MODE_CAPABILITIES,
    detect_execution_mode,
    get_mode_capabilities,
    ExecutionContext,
)


class TestExecutionMode:
    """Tests for ExecutionMode enum."""

    def test_mode_values(self) -> None:
        """Test mode enum values."""
        assert ExecutionMode.REMOTE.value == "remote"
        assert ExecutionMode.INCLUSTER.value == "incluster"
        assert ExecutionMode.DRY_RUN.value == "dry-run"

    def test_mode_from_string(self) -> None:
        """Test creating mode from string."""
        assert ExecutionMode("remote") == ExecutionMode.REMOTE
        assert ExecutionMode("incluster") == ExecutionMode.INCLUSTER
        assert ExecutionMode("dry-run") == ExecutionMode.DRY_RUN


class TestModeCapabilities:
    """Tests for ModeCapabilities dataclass."""

    def test_remote_capabilities(self) -> None:
        """Test remote mode capabilities."""
        caps = MODE_CAPABILITIES[ExecutionMode.REMOTE]
        assert caps.can_exec_pods is True
        assert caps.can_read_logs is True
        assert caps.can_call_apis is True
        assert caps.network_latency == "high"

    def test_incluster_capabilities(self) -> None:
        """Test in-cluster mode capabilities."""
        caps = MODE_CAPABILITIES[ExecutionMode.INCLUSTER]
        assert caps.can_exec_pods is True
        assert caps.can_read_logs is True
        assert caps.can_call_apis is True
        assert caps.network_latency == "low"

    def test_dry_run_capabilities(self) -> None:
        """Test dry-run mode capabilities."""
        caps = MODE_CAPABILITIES[ExecutionMode.DRY_RUN]
        assert caps.can_exec_pods is False
        assert caps.can_read_logs is False
        assert caps.can_call_apis is False
        assert caps.network_latency == "none"


class TestGetModeCapabilities:
    """Tests for get_mode_capabilities function."""

    def test_get_remote_capabilities(self) -> None:
        """Test getting remote capabilities."""
        caps = get_mode_capabilities(ExecutionMode.REMOTE)
        assert caps.can_exec_pods is True

    def test_get_dry_run_capabilities(self) -> None:
        """Test getting dry-run capabilities."""
        caps = get_mode_capabilities(ExecutionMode.DRY_RUN)
        assert caps.can_exec_pods is False


class TestDetectExecutionMode:
    """Tests for execution mode detection."""

    def test_detect_incluster_with_service_account(self) -> None:
        """Test in-cluster detection via service account token."""
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True
            mode = detect_execution_mode()
            assert mode == ExecutionMode.INCLUSTER

    def test_detect_incluster_with_env_vars(self) -> None:
        """Test in-cluster detection via Kubernetes env vars."""
        with patch("os.path.exists", return_value=False):
            with patch.dict(os.environ, {
                "KUBERNETES_SERVICE_HOST": "10.0.0.1",
            }):
                mode = detect_execution_mode()
                assert mode == ExecutionMode.INCLUSTER

    def test_detect_remote_default(self) -> None:
        """Test remote mode is default when not in-cluster."""
        with patch("os.path.exists", return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                mode = detect_execution_mode()
                assert mode == ExecutionMode.REMOTE


class TestExecutionContext:
    """Tests for ExecutionContext class."""

    def test_context_detect_remote(self) -> None:
        """Test context detection for remote mode."""
        with patch("os.path.exists", return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                ctx = ExecutionContext.detect()
                assert ctx.mode == ExecutionMode.REMOTE
                assert ctx.capabilities.can_exec_pods is True

    def test_context_with_override_dry_run(self) -> None:
        """Test context with dry-run override."""
        ctx = ExecutionContext.detect(mode_override="dry-run")
        assert ctx.mode == ExecutionMode.DRY_RUN
        assert ctx.capabilities.can_exec_pods is False
        assert ctx.explicit_override is True

    def test_context_with_override_incluster(self) -> None:
        """Test context with incluster override."""
        ctx = ExecutionContext.detect(mode_override="incluster")
        assert ctx.mode == ExecutionMode.INCLUSTER
        assert ctx.explicit_override is True

    def test_context_invalid_override(self) -> None:
        """Test invalid override falls back to detection."""
        with patch("os.path.exists", return_value=False):
            with patch.dict(os.environ, {}, clear=True):
                ctx = ExecutionContext.detect(mode_override="invalid")
                assert ctx.mode == ExecutionMode.REMOTE
                assert ctx.explicit_override is False

    def test_should_collect_runtime_metrics(self) -> None:
        """Test runtime metrics collection check."""
        remote_ctx = ExecutionContext.detect(mode_override="remote")
        assert remote_ctx.should_collect_runtime_metrics() is True

        dry_run_ctx = ExecutionContext.detect(mode_override="dry-run")
        assert dry_run_ctx.should_collect_runtime_metrics() is False

    def test_should_exec_pods(self) -> None:
        """Test pod exec check."""
        remote_ctx = ExecutionContext.detect(mode_override="remote")
        assert remote_ctx.should_exec_pods() is True

        dry_run_ctx = ExecutionContext.detect(mode_override="dry-run")
        assert dry_run_ctx.should_exec_pods() is False


class TestModeIntegration:
    """Integration tests for execution modes."""

    def test_mode_affects_capabilities(self) -> None:
        """Test different modes have different capabilities."""
        remote_ctx = ExecutionContext.detect(mode_override="remote")
        dry_run_ctx = ExecutionContext.detect(mode_override="dry-run")

        # Remote should allow exec
        assert remote_ctx.capabilities.can_exec_pods is True

        # Dry-run should not allow exec
        assert dry_run_ctx.capabilities.can_exec_pods is False

    def test_network_latency_varies(self) -> None:
        """Test network latency setting varies by mode."""
        remote_caps = get_mode_capabilities(ExecutionMode.REMOTE)
        incluster_caps = get_mode_capabilities(ExecutionMode.INCLUSTER)

        assert remote_caps.network_latency == "high"
        assert incluster_caps.network_latency == "low"
