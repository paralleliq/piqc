"""
Integration tests for framework discovery.

Tests the complete discovery pipeline against realistic
mock deployments in a kind cluster.
"""

import pytest

from piqc.core.discovery import FrameworkDetector, DeploymentDiscovery


@pytest.mark.integration
class TestFrameworkDiscoveryIntegration:
    """Integration tests for framework detection against real pods."""
    
    def test_discover_vllm_deployments(self, k8s_client, wait_for_cluster):
        """Test discovery of vLLM deployments."""
        discovery = DeploymentDiscovery()
        
        # Get pods from inference namespace
        pods = k8s_client.list_pods(
            namespace="inference",
            field_selector="status.phase=Running",
        )
        
        # Find vLLM pods
        vllm_deployments = []
        for pod in pods:
            deployment = discovery.analyze_pod(pod)
            if deployment and deployment.framework == "vllm":
                vllm_deployments.append(deployment)
        
        # Should find at least 2 vLLM deployments (llama-70b and mistral-7b)
        assert len(vllm_deployments) >= 2
        
        # Verify names
        names = {d.name for d in vllm_deployments}
        assert "vllm-llama-70b" in names or any("llama" in n for n in names)
        assert "vllm-mistral-7b" in names or any("mistral" in n for n in names)
    
    def test_discover_triton_deployment(self, k8s_client, wait_for_cluster):
        """Test discovery of Triton deployment."""
        discovery = DeploymentDiscovery()
        
        pods = k8s_client.list_pods(
            namespace="inference",
            field_selector="status.phase=Running",
        )
        
        triton_deployments = []
        for pod in pods:
            deployment = discovery.analyze_pod(pod)
            if deployment and deployment.framework == "triton":
                triton_deployments.append(deployment)
        
        # Should find Triton deployment
        assert len(triton_deployments) >= 1
    
    def test_discover_tgi_deployment(self, k8s_client, wait_for_cluster):
        """Test discovery of TGI deployment."""
        discovery = DeploymentDiscovery()
        
        pods = k8s_client.list_pods(
            namespace="inference",
            field_selector="status.phase=Running",
        )
        
        tgi_deployments = []
        for pod in pods:
            deployment = discovery.analyze_pod(pod)
            if deployment and deployment.framework == "tgi":
                tgi_deployments.append(deployment)
        
        # Should find TGI deployment
        assert len(tgi_deployments) >= 1
    
    def test_no_detection_for_nginx(self, k8s_client, wait_for_cluster):
        """Test that nginx pods are not detected as inference."""
        discovery = DeploymentDiscovery()
        
        pods = k8s_client.list_pods(
            namespace="web",
            field_selector="status.phase=Running",
        )
        
        # None of the nginx pods should be detected
        for pod in pods:
            deployment = discovery.analyze_pod(pod)
            assert deployment is None, f"Nginx pod incorrectly detected: {pod.metadata.name}"
    
    def test_confidence_scores(self, k8s_client, wait_for_cluster):
        """Test that confidence scores are reasonable."""
        discovery = DeploymentDiscovery()
        
        pods = k8s_client.list_pods(
            namespace="inference",
            field_selector="status.phase=Running",
        )
        
        for pod in pods:
            deployment = discovery.analyze_pod(pod)
            if deployment and deployment.framework != "unknown":
                # Known frameworks should have confidence >= 0.3
                assert deployment.confidence >= 0.3, (
                    f"Low confidence for {deployment.name}: {deployment.confidence}"
                )
                
                # vLLM pods with full signals should have high confidence
                if deployment.framework == "vllm" and "llama" in deployment.name.lower():
                    assert deployment.confidence >= 0.5


@pytest.mark.integration
class TestDeploymentGrouping:
    """Test deployment grouping with real replicas."""
    
    def test_group_vllm_replicas(self, k8s_client, wait_for_cluster):
        """Test that vLLM replicas are grouped correctly."""
        discovery = DeploymentDiscovery()
        
        pods = k8s_client.list_pods(
            namespace="inference",
            field_selector="status.phase=Running",
        )
        
        # Analyze all pods
        deployments = []
        for pod in pods:
            deployment = discovery.analyze_pod(pod)
            if deployment:
                deployments.append(deployment)
        
        # Group them
        grouped = discovery.group_deployments(deployments)
        
        # Find llama deployment
        llama_deployments = [d for d in grouped if "llama" in d.name.lower()]
        
        if llama_deployments:
            llama = llama_deployments[0]
            # Should have 2 replicas per the manifest
            assert llama.replicas == 2
            assert len(llama.pod_names) == 2
