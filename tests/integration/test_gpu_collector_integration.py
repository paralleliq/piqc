"""
Integration tests for GPU metrics collection.

Tests nvidia-smi execution and parsing against mock containers.
"""

import pytest

from piqc.collectors.gpu_collector import GPUCollector


@pytest.mark.integration
class TestGPUCollectorIntegration:
    """Integration tests for GPU metrics collection."""
    
    def test_collect_gpu_metrics_vllm(self, k8s_client, wait_for_cluster):
        """Test GPU metrics collection from vLLM pod."""
        gpu_collector = GPUCollector(k8s_client, exec_timeout=10)
        
        # Get a vLLM pod
        pods = k8s_client.list_pods(
            namespace="inference",
            label_selector="app=vllm-llama-70b",
        )
        
        if not pods:
            pytest.skip("No vLLM pods available")
        
        pod = pods[0]
        pod_name = pod.metadata.name
        
        # Collect metrics
        metrics = gpu_collector.collect(
            pod_name=pod_name,
            namespace="inference",
        )
        
        # Should get 8 GPUs per manifest
        assert metrics is not None
        assert len(metrics) == 8
        
        # Verify metrics structure
        for gpu in metrics:
            assert gpu.gpu_index >= 0
            assert gpu.gpu_model == "A100-SXM4-80GB"
            assert gpu.memory_total_mb == 81920
            assert gpu.memory_used_mb > 0
            assert 0 <= gpu.utilization_percent <= 100
            assert gpu.temperature_celsius > 0
    
    def test_collect_gpu_metrics_triton(self, k8s_client, wait_for_cluster):
        """Test GPU metrics collection from Triton pod."""
        gpu_collector = GPUCollector(k8s_client, exec_timeout=10)
        
        pods = k8s_client.list_pods(
            namespace="inference",
            label_selector="app=triton-server",
        )
        
        if not pods:
            pytest.skip("No Triton pods available")
        
        pod = pods[0]
        
        metrics = gpu_collector.collect(
            pod_name=pod.metadata.name,
            namespace="inference",
        )
        
        # Should get 4 GPUs per manifest
        assert metrics is not None
        assert len(metrics) == 4
        
        # Verify A100-40GB
        for gpu in metrics:
            assert gpu.gpu_model == "A100-SXM4-40GB"
            assert gpu.memory_total_mb == 40960
    
    def test_collect_gpu_metrics_tgi(self, k8s_client, wait_for_cluster):
        """Test GPU metrics collection from TGI pod."""
        gpu_collector = GPUCollector(k8s_client, exec_timeout=10)
        
        pods = k8s_client.list_pods(
            namespace="inference",
            label_selector="app=tgi-falcon",
        )
        
        if not pods:
            pytest.skip("No TGI pods available")
        
        pod = pods[0]
        
        metrics = gpu_collector.collect(
            pod_name=pod.metadata.name,
            namespace="inference",
        )
        
        # Should get 2 GPUs per manifest
        assert metrics is not None
        assert len(metrics) == 2
        
        # Verify A10G
        for gpu in metrics:
            assert gpu.gpu_model == "A10G"
            assert gpu.memory_total_mb == 24576
    
    def test_gpu_metrics_variation(self, k8s_client, wait_for_cluster):
        """Test that GPU metrics have realistic variation."""
        gpu_collector = GPUCollector(k8s_client, exec_timeout=10)
        
        pods = k8s_client.list_pods(
            namespace="inference",
            label_selector="app=vllm-llama-70b",
        )
        
        if not pods:
            pytest.skip("No vLLM pods available")
        
        pod = pods[0]
        
        metrics = gpu_collector.collect(
            pod_name=pod.metadata.name,
            namespace="inference",
        )
        
        if metrics and len(metrics) > 1:
            # Metrics should have some variation per GPU
            utilizations = [m.utilization_percent for m in metrics]
            temperatures = [m.temperature_celsius for m in metrics]
            
            # Not all exactly the same (our fake script adds variation)
            assert len(set(utilizations)) > 1 or len(metrics) == 1
            assert len(set(temperatures)) > 1 or len(metrics) == 1
