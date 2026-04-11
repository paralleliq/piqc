"""Unit tests for GPU fragmentation detection logic."""

import pytest
from unittest.mock import MagicMock

from piqc.core.orchestrator import (
    FragmentedNodeInfo,
    ScanOrchestrator,
    UnallocatedNodeInfo,
)
from piqc.models.modelspec import ModelSpec, ResourceInfo


def _make_modelspec(gpu_count: int, model_name: str = "test-model") -> ModelSpec:
    """Build a minimal ModelSpec with a given GPU count."""
    spec = MagicMock(spec=ModelSpec)
    spec.resources = MagicMock(spec=ResourceInfo)
    spec.resources.gpus = [MagicMock() for _ in range(gpu_count)]
    spec.model = MagicMock()
    spec.model.name = model_name
    return spec


def _make_orchestrator() -> ScanOrchestrator:
    k8s_client = MagicMock()
    return ScanOrchestrator(k8s_client=k8s_client)


def _make_unallocated(node_name: str, total: int, allocated: int, gpu_type: str = "nvidia-l4") -> UnallocatedNodeInfo:
    return UnallocatedNodeInfo(
        node_name=node_name,
        gpu_type=gpu_type,
        total_gpus=total,
        allocated_gpus=allocated,
        unallocated_gpus=total - allocated,
    )


class TestAnalyzeFragmentation:

    def test_no_unallocated_nodes_returns_empty(self):
        orch = _make_orchestrator()
        modelspecs = [_make_modelspec(gpu_count=1)]
        result = orch._analyze_fragmentation([], modelspecs)
        assert result == []

    def test_no_modelspecs_returns_empty(self):
        orch = _make_orchestrator()
        unallocated = [_make_unallocated("node-a", total=8, allocated=7)]
        result = orch._analyze_fragmentation(unallocated, [])
        assert result == []

    def test_stranded_slot_detected(self):
        """Node with 1 free GPU, smallest model needs 2 → stranded."""
        orch = _make_orchestrator()
        unallocated = [_make_unallocated("node-a", total=8, allocated=7)]
        modelspecs = [_make_modelspec(gpu_count=2)]

        result = orch._analyze_fragmentation(unallocated, modelspecs)

        assert len(result) == 1
        assert result[0].node_name == "node-a"
        assert result[0].stranded_gpus == 1
        assert result[0].min_model_gpus_needed == 2

    def test_sufficient_free_slots_not_flagged(self):
        """Node with 2 free GPUs, smallest model needs 1 → not stranded."""
        orch = _make_orchestrator()
        unallocated = [_make_unallocated("node-a", total=8, allocated=6)]
        modelspecs = [_make_modelspec(gpu_count=1)]

        result = orch._analyze_fragmentation(unallocated, modelspecs)

        assert result == []

    def test_exact_fit_not_flagged(self):
        """Node with exactly enough free GPUs for smallest model → not stranded."""
        orch = _make_orchestrator()
        unallocated = [_make_unallocated("node-a", total=8, allocated=6)]
        modelspecs = [_make_modelspec(gpu_count=2)]

        result = orch._analyze_fragmentation(unallocated, modelspecs)

        assert result == []

    def test_multiple_nodes_mixed(self):
        """Two nodes: one stranded, one sufficient."""
        orch = _make_orchestrator()
        unallocated = [
            _make_unallocated("node-a", total=8, allocated=7),  # 1 free — stranded
            _make_unallocated("node-b", total=8, allocated=6),  # 2 free — ok
        ]
        modelspecs = [_make_modelspec(gpu_count=2)]

        result = orch._analyze_fragmentation(unallocated, modelspecs)

        assert len(result) == 1
        assert result[0].node_name == "node-a"

    def test_min_gpu_footprint_used(self):
        """Uses the smallest model GPU count as the fragmentation threshold."""
        orch = _make_orchestrator()
        unallocated = [_make_unallocated("node-a", total=8, allocated=6)]  # 2 free
        modelspecs = [
            _make_modelspec(gpu_count=4),  # large model
            _make_modelspec(gpu_count=2),  # small model — threshold should be 2
        ]

        # 2 free >= 2 min → not stranded
        result = orch._analyze_fragmentation(unallocated, modelspecs)
        assert result == []

    def test_stranded_gpu_count_and_type_preserved(self):
        """FragmentedNodeInfo carries correct stranded count and GPU type."""
        orch = _make_orchestrator()
        unallocated = [_make_unallocated("node-x", total=4, allocated=3, gpu_type="H100-SXM4-80GB")]
        modelspecs = [_make_modelspec(gpu_count=2)]

        result = orch._analyze_fragmentation(unallocated, modelspecs)

        assert len(result) == 1
        node = result[0]
        assert node.stranded_gpus == 1
        assert node.gpu_type == "H100-SXM4-80GB"
        assert node.total_gpus == 4
        assert node.allocated_gpus == 3
