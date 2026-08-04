"""
Pytest configuration and fixtures for real-GPU hardware verification tests.

Unlike tests/integration/ (which verifies piqc's collection *mechanism* against
a kind cluster with mock containers — no real GPU needed), tests in this
directory deploy a deliberately-engineered scenario onto a real GPU node and
confirm piqc reports facts that match reality (e.g., a genuine OOM kill, a
genuine tier mismatch, genuine KV cache saturation). These are the only piqc
tests that cost real GPU-hours to run.

Scales with the size of the rule catalog and the GPU/framework combinations
it needs to support -- not with customer count. See
strategy/... (GTM/margin discussion, 2026-08) for the reasoning behind that
distinction.
"""

import os

import pytest

# Check if hardware tests should run
HARDWARE_TESTS_ENABLED = os.environ.get("PARALLELIQ_HARDWARE_TESTS", "0") == "1"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "hardware: mark test as requiring a real GPU node (not kind/mocked)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip hardware tests unless explicitly enabled."""
    if HARDWARE_TESTS_ENABLED:
        return

    skip_hardware = pytest.mark.skip(
        reason=(
            "Real-GPU hardware tests disabled. Set PARALLELIQ_HARDWARE_TESTS=1 "
            "and PARALLELIQ_HARDWARE_KUBECONFIG to point at a cluster with a "
            "real GPU node to enable."
        )
    )

    for item in items:
        if "hardware" in item.keywords:
            item.add_marker(skip_hardware)


@pytest.fixture(scope="session")
def hardware_kubeconfig() -> str:
    """Path to the kubeconfig for the real-GPU verification cluster.

    Deliberately separate from the kind cluster's kubeconfig used by
    tests/integration/ -- this must point at a cluster with an actual
    GPU node (a cheap spot T4/L4 is enough for most scenarios; see the
    per-rule scenario table in each test module's docstring).
    """
    path = os.environ.get("PARALLELIQ_HARDWARE_KUBECONFIG")
    if not path:
        pytest.skip("PARALLELIQ_HARDWARE_KUBECONFIG not set")
    return path
