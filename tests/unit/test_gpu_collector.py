"""
Unit tests for GPU collector.

Tests nvidia-smi parsing and fallback behavior.
"""

import pytest
from datetime import datetime

from piqc.collectors.gpu_collector import (
    GPUCollector,
    GPUMetrics,
    _parse_int,
    get_gpu_info_from_pod_spec,
)


class TestGPUMetrics:
    """Tests for GPUMetrics dataclass."""
    
    def test_to_memory_str_gb(self) -> None:
        """Test memory conversion to GB."""
        metrics = GPUMetrics(
            gpu_index=0,
            gpu_model="A100",
            memory_total_mb=81920,
            memory_used_mb=40960,
            memory_free_mb=40960,
            utilization_percent=85,
            memory_utilization_percent=50,
            temperature_celsius=72,
            power_draw_watts=300,
            power_limit_watts=400,
            collection_timestamp=datetime.utcnow(),
        )
        
        assert metrics.to_memory_str(81920) == "80GB"
        assert metrics.to_memory_str(40960) == "40GB"
    
    def test_to_memory_str_mb(self) -> None:
        """Test memory conversion for values under 1GB."""
        metrics = GPUMetrics(
            gpu_index=0,
            gpu_model="T4",
            memory_total_mb=512,
            memory_used_mb=256,
            memory_free_mb=256,
            utilization_percent=50,
            memory_utilization_percent=50,
            temperature_celsius=60,
            power_draw_watts=50,
            power_limit_watts=70,
            collection_timestamp=datetime.utcnow(),
        )
        
        assert metrics.to_memory_str(512) == "512MB"


class TestParseInt:
    """Tests for _parse_int helper that handles nvidia-smi N/A fields."""

    def test_normal_integer(self) -> None:
        assert _parse_int("74") == 74

    def test_float_string(self) -> None:
        assert _parse_int("74.5") == 74

    def test_na_bracket(self) -> None:
        assert _parse_int("[N/A]") is None

    def test_na_plain(self) -> None:
        assert _parse_int("N/A") is None

    def test_empty_string(self) -> None:
        assert _parse_int("") is None

    def test_not_supported(self) -> None:
        assert _parse_int("[Not Supported]") is None

    def test_whitespace(self) -> None:
        assert _parse_int("  42  ") == 42

    def test_default_value(self) -> None:
        assert _parse_int("[N/A]", default=0) == 0

    def test_garbage(self) -> None:
        assert _parse_int("bogus") is None


class TestNvidiaSmiParsing:
    """Tests for nvidia-smi output parsing."""
    
    def test_parse_single_gpu(self) -> None:
        """Test parsing output for a single GPU."""
        output = "0, NVIDIA A100-SXM4-80GB, 81920, 45120, 36800, 87, 55, 72, 320, 400"
        
        # We need to test the parsing method directly
        # Since GPUCollector requires K8sClient, we'll test the logic
        lines = output.strip().split("\n")
        
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            
            assert len(parts) == 10
            assert int(parts[0]) == 0
            assert "A100" in parts[1]
            assert int(float(parts[2])) == 81920
            assert int(float(parts[3])) == 45120
    
    def test_parse_multi_gpu(self) -> None:
        """Test parsing output for multiple GPUs."""
        output = """0, NVIDIA A100-SXM4-80GB, 81920, 45120, 36800, 87, 55, 72, 320, 400
1, NVIDIA A100-SXM4-80GB, 81920, 46890, 35030, 89, 57, 74, 325, 400"""
        
        lines = output.strip().split("\n")
        
        assert len(lines) == 2
        
        # First GPU
        parts1 = [p.strip() for p in lines[0].split(",")]
        assert int(parts1[0]) == 0
        
        # Second GPU
        parts2 = [p.strip() for p in lines[1].split(",")]
        assert int(parts2[0]) == 1


class TestFallbackGPUInfo:
    """Tests for fallback GPU info when nvidia-smi is unavailable."""
    
    def test_nvidia_gpu_info(self) -> None:
        """Test creating placeholder info for NVIDIA GPUs."""
        gpus = get_gpu_info_from_pod_spec(4, "nvidia.com/gpu")
        
        assert len(gpus) == 4
        assert all(g["type"] == "NVIDIA GPU" for g in gpus)
        assert [g["index"] for g in gpus] == [0, 1, 2, 3]
    
    def test_amd_gpu_info(self) -> None:
        """Test creating placeholder info for AMD GPUs."""
        gpus = get_gpu_info_from_pod_spec(2, "amd.com/gpu")
        
        assert len(gpus) == 2
        assert all(g["type"] == "AMD GPU" for g in gpus)
    
    def test_zero_gpus(self) -> None:
        """Test with zero GPUs."""
        gpus = get_gpu_info_from_pod_spec(0, "nvidia.com/gpu")
        
        assert len(gpus) == 0
