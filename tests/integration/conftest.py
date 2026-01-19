"""
Pytest configuration and fixtures for integration tests.

Provides fixtures for managing the test Kubernetes cluster
and running tests against realistic mock deployments.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Generator

import pytest

# Check if integration tests should run
INTEGRATION_ENABLED = os.environ.get("MODELSPEC_INTEGRATION_TESTS", "0") == "1"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test requiring kind cluster",
    )


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if not enabled."""
    if INTEGRATION_ENABLED:
        return
    
    skip_integration = pytest.mark.skip(
        reason="Integration tests disabled. Set MODELSPEC_INTEGRATION_TESTS=1 to enable."
    )
    
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def cluster_context() -> str:
    """
    Get the test cluster context name.
    
    Returns:
        Kubernetes context for the test cluster.
    """
    return "kind-modelspec-test"


@pytest.fixture(scope="session")
def wait_for_cluster(cluster_context: str) -> Generator[str, None, None]:
    """
    Wait for the test cluster to be ready.
    
    This fixture ensures the cluster is accessible before
    running integration tests.
    """
    if not INTEGRATION_ENABLED:
        pytest.skip("Integration tests disabled")
    
    # Check if cluster exists
    result = subprocess.run(
        ["kubectl", "cluster-info", "--context", cluster_context],
        capture_output=True,
        text=True,
    )
    
    if result.returncode != 0:
        pytest.skip(f"Test cluster not available: {result.stderr}")
    
    # Wait for pods to be ready
    max_wait = 60
    start = time.time()
    
    while time.time() - start < max_wait:
        result = subprocess.run(
            [
                "kubectl", "get", "pods",
                "-n", "inference",
                "-o", "jsonpath={.items[*].status.phase}",
                "--context", cluster_context,
            ],
            capture_output=True,
            text=True,
        )
        
        phases = result.stdout.split()
        if phases and all(p == "Running" for p in phases):
            break
        
        time.sleep(2)
    
    yield cluster_context


@pytest.fixture(scope="session")
def k8s_client(wait_for_cluster: str):
    """
    Create a K8sClient connected to the test cluster.
    
    Returns:
        Configured K8sClient instance.
    """
    from piqc.core.k8s_client import K8sClient
    
    # Use the test cluster context
    client = K8sClient(context=wait_for_cluster)
    return client


@pytest.fixture(scope="session")
def orchestrator(k8s_client):
    """
    Create an orchestrator for the test cluster.
    
    Returns:
        Configured ScanOrchestrator instance.
    """
    from piqc.core.orchestrator import ScanOrchestrator
    
    return ScanOrchestrator(
        k8s_client=k8s_client,
        enable_exec=True,
        enable_logs=True,
        workers=3,
        timeout=30,
    )


@pytest.fixture
def test_manifests_dir() -> Path:
    """Get the path to test manifests directory."""
    return Path(__file__).parent / "kind" / "manifests"
