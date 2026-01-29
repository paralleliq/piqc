"""
Execution mode detection and configuration.

Detects whether the tool is running in-cluster, remotely,
or in dry-run mode, and adapts capabilities accordingly.
"""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from piqc.utils.logger import get_logger


logger = get_logger(__name__)


class ExecutionMode(Enum):
    """Execution context modes."""

    REMOTE = "remote"
    INCLUSTER = "incluster"
    DRY_RUN = "dry-run"


@dataclass
class ModeCapabilities:
    """
    Capabilities available in each execution mode.

    Defines what operations are available based on
    the execution context.
    """

    can_exec_pods: bool
    can_read_logs: bool
    can_call_apis: bool
    network_latency: str  # "low", "medium", "high"


# Mode capability definitions
MODE_CAPABILITIES = {
    ExecutionMode.REMOTE: ModeCapabilities(
        can_exec_pods=True,  # If RBAC permits
        can_read_logs=True,  # If RBAC permits
        can_call_apis=True,  # If services exposed
        network_latency="high",
    ),
    ExecutionMode.INCLUSTER: ModeCapabilities(
        can_exec_pods=True,
        can_read_logs=True,
        can_call_apis=True,
        network_latency="low",
    ),
    ExecutionMode.DRY_RUN: ModeCapabilities(
        can_exec_pods=False,
        can_read_logs=False,
        can_call_apis=False,
        network_latency="none",
    ),
}


def detect_execution_mode() -> ExecutionMode:
    """
    Detect the current execution mode.

    Checks for in-cluster indicators:
    1. Kubernetes service account token
    2. Kubernetes environment variables

    Returns:
        ExecutionMode.INCLUSTER if running inside K8s,
        ExecutionMode.REMOTE otherwise.
    """
    # Check for service account token (most reliable indicator)
    sa_token_path = "/var/run/secrets/kubernetes.io/serviceaccount/token"

    if os.path.exists(sa_token_path):
        logger.debug("Detected in-cluster execution (service account token present)")
        return ExecutionMode.INCLUSTER

    # Check for Kubernetes environment variables
    if os.environ.get("KUBERNETES_SERVICE_HOST"):
        logger.debug("Detected in-cluster execution (KUBERNETES_SERVICE_HOST set)")
        return ExecutionMode.INCLUSTER

    logger.debug("Detected remote execution")
    return ExecutionMode.REMOTE


def get_mode_capabilities(mode: ExecutionMode) -> ModeCapabilities:
    """
    Get capabilities for a specific execution mode.

    Args:
        mode: The execution mode.

    Returns:
        ModeCapabilities for the given mode.
    """
    return MODE_CAPABILITIES.get(mode, MODE_CAPABILITIES[ExecutionMode.REMOTE])


@dataclass
class ExecutionContext:
    """
    Complete execution context information.

    Combines mode detection with configuration overrides.
    """

    mode: ExecutionMode
    capabilities: ModeCapabilities
    explicit_override: bool = False

    @classmethod
    def detect(cls, mode_override: Optional[str] = None) -> "ExecutionContext":
        """
        Detect or create execution context.

        Args:
            mode_override: Optional mode string to override detection.

        Returns:
            ExecutionContext with appropriate capabilities.
        """
        if mode_override:
            try:
                mode = ExecutionMode(mode_override.lower())
                return cls(
                    mode=mode,
                    capabilities=get_mode_capabilities(mode),
                    explicit_override=True,
                )
            except ValueError:
                logger.warning(f"Unknown mode '{mode_override}', using auto-detection")

        mode = detect_execution_mode()
        return cls(
            mode=mode,
            capabilities=get_mode_capabilities(mode),
            explicit_override=False,
        )

    def should_collect_runtime_metrics(self) -> bool:
        """Check if runtime metrics collection is appropriate."""
        return self.capabilities.can_call_apis and self.mode != ExecutionMode.DRY_RUN

    def should_exec_pods(self) -> bool:
        """Check if pod exec operations are appropriate."""
        return self.capabilities.can_exec_pods and self.mode != ExecutionMode.DRY_RUN
