"""
End-to-end integration tests.

Tests the complete scanning pipeline from CLI to ModelSpec output.
"""

import pytest
import json
import tempfile
from pathlib import Path


@pytest.mark.integration
class TestEndToEndScan:
    """End-to-end tests for the complete scanning pipeline."""
    
    def test_full_scan(self, orchestrator, wait_for_cluster):
        """Test a full cluster scan."""
        result = orchestrator.scan(namespaces=["inference"])
        
        # Should find deployments
        assert result.deployments_found >= 4  # vllm x2, triton, tgi
        
        # Should generate modelspecs
        assert len(result.modelspecs) >= 4
        
        # Should have analyzed pods
        assert result.pods_analyzed >= 4
        
        # Scan should complete in reasonable time
        assert result.duration_seconds < 60
    
    def test_scan_specific_namespace(self, orchestrator, wait_for_cluster):
        """Test scanning a specific namespace."""
        # Scan inference namespace
        result = orchestrator.scan(namespaces=["inference"])
        
        inference_count = len(result.modelspecs)
        assert inference_count >= 4
        
        # Scan web namespace (should find nothing)
        result_web = orchestrator.scan(namespaces=["web"])
        assert len(result_web.modelspecs) == 0
    
    def test_modelspec_completeness(self, orchestrator, wait_for_cluster):
        """Test that generated ModelSpecs have complete data."""
        result = orchestrator.scan(namespaces=["inference"])
        
        for spec in result.modelspecs:
            # Required fields
            assert spec.metadata.name
            assert spec.metadata.namespace == "inference"
            assert spec.metadata.collector_version
            assert spec.metadata.collection_timestamp
            
            # Engine info
            assert spec.engine.name in ("vllm", "triton", "tgi", "unknown")
            assert 0 <= spec.engine.detection_confidence <= 1.0
            
            # Resources
            assert spec.resources.replicas >= 1
            
            # Kubernetes metadata
            assert spec.kubernetes.deployment_type in ("Deployment", "StatefulSet", "Pod")
            assert spec.kubernetes.image
            
            # Collection metadata
            assert spec.collection.mode in ("remote", "incluster")
            assert spec.collection.data_completeness is not None
    
    def test_vllm_modelspec_details(self, orchestrator, wait_for_cluster):
        """Test that vLLM ModelSpecs have expected details."""
        result = orchestrator.scan(namespaces=["inference"])
        
        # Find a vLLM modelspec
        vllm_specs = [s for s in result.modelspecs if s.engine.name == "vllm"]
        assert len(vllm_specs) >= 2
        
        # Check llama deployment
        llama_specs = [s for s in vllm_specs if "llama" in s.metadata.name.lower()]
        if llama_specs:
            llama = llama_specs[0]
            
            # Model info
            assert llama.model.name == "meta-llama/Llama-2-70b-chat-hf"
            assert llama.model.served_name == "llama-70b"
            
            # Inference config
            assert llama.inference.tensor_parallel_size == 8
            assert llama.inference.precision == "bfloat16"
            assert llama.inference.max_model_len == 4096
            
            # GPU info (if collected)
            if llama.resources.gpus:
                assert len(llama.resources.gpus) == 8
                assert llama.resources.gpus[0].type == "A100-SXM4-80GB"
    
    def test_gpu_metrics_in_output(self, orchestrator, wait_for_cluster):
        """Test that GPU metrics are included in output."""
        result = orchestrator.scan(namespaces=["inference"])
        
        # At least one spec should have GPU metrics
        specs_with_gpus = [s for s in result.modelspecs if s.resources.gpus]
        
        assert len(specs_with_gpus) >= 1, "No specs have GPU metrics"
        
        # Check GPU data completeness
        for spec in specs_with_gpus:
            assert spec.collection.data_completeness.gpu_metrics is True
            
            for gpu in spec.resources.gpus:
                assert gpu.type is not None
                assert gpu.memory_total is not None
    
    def test_yaml_output(self, orchestrator, wait_for_cluster):
        """Test YAML file generation."""
        from piqc.generators.yaml_generator import YAMLGenerator
        
        result = orchestrator.scan(namespaces=["inference"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_gen = YAMLGenerator()
            files = yaml_gen.generate_multi(result.modelspecs, tmpdir)
            
            assert len(files) == len(result.modelspecs)
            
            # Verify files exist and are valid YAML
            import yaml
            for filepath in files:
                path = Path(filepath)
                assert path.exists()
                
                with open(path) as f:
                    data = yaml.safe_load(f)
                
                assert data["apiVersion"] == "modelspec.paralleliq.ai/v1"
                assert data["kind"] == "ModelSpec"
    
    def test_json_output(self, orchestrator, wait_for_cluster):
        """Test JSON file generation."""
        from piqc.generators.json_generator import JSONGenerator
        
        result = orchestrator.scan(namespaces=["inference"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            json_gen = JSONGenerator()
            output_file = f"{tmpdir}/modelspecs.json"
            json_gen.generate_combined(result.modelspecs, output_file)
            
            # Verify file
            with open(output_file) as f:
                data = json.load(f)
            
            assert isinstance(data, list)
            assert len(data) == len(result.modelspecs)
            
            for item in data:
                assert item["apiVersion"] == "modelspec.paralleliq.ai/v1"


@pytest.mark.integration
class TestCLIIntegration:
    """Test CLI commands against real cluster."""
    
    def test_cli_test_connection(self, wait_for_cluster):
        """Test the test-connection command."""
        import subprocess
        
        result = subprocess.run(
            [
                "poetry", "run", "piqc",
                "test-connection",
                "--context", wait_for_cluster,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],  # Project root
        )
        
        assert result.returncode == 0
        assert "Connection successful" in result.stdout or "successful" in result.stdout.lower()
    
    def test_cli_scan_table(self, wait_for_cluster):
        """Test the scan command with table output."""
        import subprocess
        
        result = subprocess.run(
            [
                "poetry", "run", "piqc",
                "scan",
                "--namespace", "inference",
                "--format", "table",
                "--context", wait_for_cluster,
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parents[2],
        )
        
        assert result.returncode == 0
        # Should mention found deployments
        assert "Inference deployments found" in result.stdout or "deployment" in result.stdout.lower()
    
    def test_cli_scan_yaml(self, wait_for_cluster):
        """Test the scan command with YAML output."""
        import subprocess
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = subprocess.run(
                [
                    "poetry", "run", "piqc",
                    "scan",
                    "--namespace", "inference",
                    "--format", "yaml",
                    "--output", tmpdir,
                    "--context", wait_for_cluster,
                ],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parents[2],
            )
            
            assert result.returncode == 0
            
            # Should have generated YAML files
            yaml_files = list(Path(tmpdir).glob("*.yaml"))
            assert len(yaml_files) >= 1
