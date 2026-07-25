"""
Pydantic models for ModelSpec schema.

Defines the complete data structures for ModelSpec YAML generation
with validation, serialization, and alias support.
"""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict


class MetadataInfo(BaseModel):
    """Kubernetes-style metadata for the ModelSpec."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    name: str = Field(..., description="Deployment name")
    namespace: str = Field(..., description="Kubernetes namespace")
    uid: Optional[str] = Field(None, description="Deployment UID")
    labels: dict[str, str] = Field(default_factory=dict, description="Kubernetes labels")
    annotations: dict[str, str] = Field(default_factory=dict, description="Kubernetes annotations")
    collection_timestamp: str = Field(
        ...,
        alias="collectionTimestamp",
        description="ISO 8601 timestamp of data collection",
    )
    collector_version: str = Field(
        ...,
        alias="collectorVersion",
        description="Version of the piqc",
    )


class ModelInfo(BaseModel):
    """Information about the ML model being served."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    name: Optional[str] = Field(None, description="Model name (e.g., meta-llama/Llama-2-7b-hf)")
    served_name: Optional[str] = Field(
        None,
        alias="servedName",
        description="Name exposed via API",
    )
    source: Optional[str] = Field(
        None,
        description="Model source (huggingface, s3, gcs, local)",
    )
    source_path: Optional[str] = Field(
        None,
        alias="sourcePath",
        description="Path or URL to model",
    )
    architecture: Optional[str] = Field(
        None,
        description="Model architecture (llama, gpt, falcon, etc.)",
    )
    parameters: Optional[str] = Field(
        None,
        description="Parameter count (7B, 13B, 70B, etc.)",
    )
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Model identification confidence",
    )
    identification_method: Optional[str] = Field(
        None,
        alias="identificationMethod",
        description="How the model was identified",
    )


class EngineInfo(BaseModel):
    """Information about the inference engine."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    name: str = Field(..., description="Engine name (vllm, triton, tgi, unknown)")
    version: Optional[str] = Field(None, description="Engine version")
    framework: Optional[str] = Field(None, description="Python framework if detected")
    detection_confidence: float = Field(
        0.0,
        alias="detectionConfidence",
        ge=0.0,
        le=1.0,
        description="Framework detection confidence",
    )


class InferenceConfig(BaseModel):
    """Inference configuration settings."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    precision: Optional[str] = Field(
        None,
        description="Precision (float16, bfloat16, float32, int8)",
    )
    quantization: Optional[str] = Field(
        None,
        description="Quantization method (awq, gptq, squeezellm)",
    )
    max_model_len: Optional[int] = Field(
        None,
        alias="maxModelLen",
        description="Maximum context length",
    )
    max_batch_tokens: Optional[int] = Field(
        None,
        alias="maxBatchTokens",
        description="Maximum tokens in batch",
    )
    max_sequences: Optional[int] = Field(
        None,
        alias="maxSequences",
        description="Maximum concurrent sequences",
    )
    tensor_parallel_size: Optional[int] = Field(
        None,
        alias="tensorParallelSize",
        description="GPU parallelism for tensor operations",
    )
    pipeline_parallel_size: Optional[int] = Field(
        None,
        alias="pipelineParallelSize",
        description="Pipeline parallelism",
    )


class GPUInfo(BaseModel):
    """Information about a single GPU."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    index: int = Field(..., description="GPU index")
    type: str = Field(..., description="GPU type (e.g., A100-SXM4-80GB)")
    memory_total: str = Field(..., alias="memoryTotal", description="Total GPU memory")
    memory_used: Optional[str] = Field(None, alias="memoryUsed", description="Used GPU memory")
    utilization: Optional[int] = Field(
        None,
        ge=0,
        le=100,
        description="GPU utilization percentage",
    )
    temperature: Optional[int] = Field(None, description="GPU temperature in Celsius")
    power_draw: Optional[int] = Field(None, alias="powerDraw", description="Power draw in Watts")
    node: Optional[str] = Field(None, description="Kubernetes node name")
    pod_name: Optional[str] = Field(None, alias="podName", description="Pod name")


class CPUInfo(BaseModel):
    """Real-time CPU utilization for a deployment, sampled via /proc/stat."""

    model_config = ConfigDict(populate_by_name=True)

    utilization_percent: float = Field(
        ...,
        alias="utilizationPercent",
        ge=0.0,
        le=100.0,
        description="CPU utilization percentage",
    )
    iowait_percent: float = Field(
        ...,
        alias="iowaitPercent",
        ge=0.0,
        le=100.0,
        description="Percentage of time CPU was idle while waiting on I/O",
    )
    cpu_count: Optional[int] = Field(None, alias="cpuCount", description="Number of CPU cores observed")
    pod_name: Optional[str] = Field(None, alias="podName", description="Pod name")


class ResourceInfo(BaseModel):
    """Resource allocation information."""

    model_config = ConfigDict(populate_by_name=True)

    replicas: int = Field(..., description="Number of pod replicas")
    gpus: list[GPUInfo] = Field(default_factory=list, description="GPU information")
    cpu: Optional[CPUInfo] = Field(None, description="Real-time CPU utilization")
    gpu_memory_utilization: Optional[float] = Field(
        None,
        alias="gpuMemoryUtilization",
        ge=0.0,
        le=1.0,
        description="vLLM GPU memory utilization setting",
    )
    cpu_request: Optional[str] = Field(None, alias="cpuRequest", description="CPU request")
    cpu_limit: Optional[str] = Field(None, alias="cpuLimit", description="CPU limit")
    memory_request: Optional[str] = Field(None, alias="memoryRequest", description="Memory request")
    memory_limit: Optional[str] = Field(None, alias="memoryLimit", description="Memory limit")


class Endpoint(BaseModel):
    """Network endpoint information."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    type: str = Field(..., description="Endpoint type (http, grpc)")
    url: Optional[str] = Field(None, description="Service URL")
    port: int = Field(..., description="Port number")
    path: Optional[str] = Field(None, description="API path (e.g., /v1/completions)")


class KubernetesMetadata(BaseModel):
    """Kubernetes-specific metadata."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    deployment_type: str = Field(
        ...,
        alias="deploymentType",
        description="Workload type (Deployment, StatefulSet)",
    )
    cluster_name: Optional[str] = Field(
        None,
        alias="clusterName",
        description="Cluster name if available",
    )
    creation_timestamp: Optional[str] = Field(
        None,
        alias="creationTimestamp",
        description="Workload creation timestamp",
    )
    image: str = Field(..., description="Container image")
    image_tag: Optional[str] = Field(None, alias="imageTag", description="Image tag")


class DataCompleteness(BaseModel):
    """Indicates which data was successfully collected."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    static_config: bool = Field(
        True,
        alias="staticConfig",
        description="Static configuration collected",
    )
    gpu_metrics: bool = Field(
        False,
        alias="gpuMetrics",
        description="GPU metrics collected",
    )
    cpu_metrics: bool = Field(
        False,
        alias="cpuMetrics",
        description="CPU metrics collected",
    )
    runtime_metrics: bool = Field(
        False,
        alias="runtimeMetrics",
        description="Runtime metrics collected",
    )


class VLLMRuntimeState(BaseModel):
    """Runtime state from vLLM API."""

    model_config = ConfigDict(populate_by_name=True)

    loaded_model: Optional[str] = Field(None, alias="loadedModel", description="Currently loaded model")

    # Request metrics
    requests_running: int = Field(0, alias="requestsRunning", description="Requests currently processing")
    requests_waiting: int = Field(0, alias="requestsWaiting", description="Requests in queue")
    requests_swapped: int = Field(0, alias="requestsSwapped", description="Requests swapped to CPU")

    # Latency metrics (seconds)
    ttft_p50: Optional[float] = Field(None, alias="ttftP50", description="Time to First Token p50")
    ttft_p95: Optional[float] = Field(None, alias="ttftP95", description="Time to First Token p95")
    ttft_p99: Optional[float] = Field(None, alias="ttftP99", description="Time to First Token p99")
    tpot_p50: Optional[float] = Field(None, alias="tpotP50", description="Time Per Output Token p50")
    tpot_p95: Optional[float] = Field(None, alias="tpotP95", description="Time Per Output Token p95")
    e2e_p50: Optional[float] = Field(None, alias="e2eP50", description="End-to-end latency p50")
    e2e_p95: Optional[float] = Field(None, alias="e2eP95", description="End-to-end latency p95")
    e2e_p99: Optional[float] = Field(None, alias="e2eP99", description="End-to-end latency p99")

    # Throughput metrics
    prompt_tokens_per_sec: Optional[float] = Field(
        None, alias="promptTokensPerSec", description="Prompt token throughput"
    )
    generation_tokens_per_sec: Optional[float] = Field(
        None, alias="generationTokensPerSec", description="Generation token throughput"
    )
    prompt_tokens_total: Optional[int] = Field(
        None, alias="promptTokensTotal", description="Total prompt tokens processed"
    )
    generation_tokens_total: Optional[int] = Field(
        None, alias="generationTokensTotal", description="Total tokens generated"
    )

    # Cache metrics
    gpu_cache_usage_percent: float = Field(
        0.0, alias="gpuCacheUsagePercent", description="GPU KV cache usage %"
    )
    cpu_cache_usage_percent: float = Field(
        0.0, alias="cpuCacheUsagePercent", description="CPU KV cache usage %"
    )
    prefix_cache_hit_rate: float = Field(
        0.0, alias="prefixCacheHitRate", description="Prefix cache hit rate"
    )

    # Status
    api_available: bool = Field(False, alias="apiAvailable", description="vLLM API accessible")
    health_status: str = Field("unknown", alias="healthStatus", description="Health check status")
    collection_timestamp: Optional[str] = Field(
        None, alias="collectionTimestamp", description="Metrics collection time"
    )


class RuntimeState(BaseModel):
    """Runtime state information from API endpoints."""

    model_config = ConfigDict(populate_by_name=True)

    vllm: Optional[VLLMRuntimeState] = Field(None, description="vLLM-specific runtime state")
    collection_method: str = Field(
        "kubernetes-only",
        alias="collectionMethod",
        description="Collection method (kubernetes-only, vllm-api)",
    )


class CollectionMetadata(BaseModel):
    """Metadata about the collection process."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    mode: str = Field(..., description="Collection mode (remote, incluster)")
    duration: str = Field(..., description="Scan duration")
    warnings: list[str] = Field(default_factory=list, description="Collection warnings")
    errors: list[str] = Field(default_factory=list, description="Collection errors")
    data_completeness: DataCompleteness = Field(
        ...,
        alias="dataCompleteness",
        description="Data completeness indicators",
    )


class AutoscalingInfo(BaseModel):
    """Autoscaling configuration from HPA."""
    
    model_config = ConfigDict(populate_by_name=True)
    
    enabled: bool = Field(False, description="Whether HPA is enabled")
    metric_type: Optional[str] = Field(
        None,
        alias="metricType",
        description="Primary HPA metric type (cpu, memory, custom)",
    )
    min_replicas: Optional[int] = Field(None, alias="minReplicas")
    max_replicas: Optional[int] = Field(None, alias="maxReplicas")
    current_replicas: Optional[int] = Field(None, alias="currentReplicas")


class ModelSpec(BaseModel):
    """
    Complete ModelSpec document.
    
    This is the root model for generating ModelSpec YAML files.
    It contains all information about a discovered ML model deployment.
    """
    
    model_config = ConfigDict(populate_by_name=True)
    
    api_version: str = Field(
        "modelspec.paralleliq.ai/v1",
        alias="apiVersion",
        description="ModelSpec API version",
    )
    kind: str = Field("ModelSpec", description="Resource kind")
    metadata: MetadataInfo = Field(..., description="Resource metadata")
    model: ModelInfo = Field(..., description="Model information")
    engine: EngineInfo = Field(..., description="Inference engine information")
    inference: InferenceConfig = Field(..., description="Inference configuration")
    resources: ResourceInfo = Field(..., description="Resource allocation")
    endpoints: list[Endpoint] = Field(default_factory=list, description="Network endpoints")
    kubernetes: KubernetesMetadata = Field(..., description="Kubernetes metadata")
    runtime_state: Optional[RuntimeState] = Field(
        None,
        alias="runtimeState",
        description="Runtime state from API endpoints",
    )
    collection: CollectionMetadata = Field(..., description="Collection metadata")
    autoscaling: Optional[AutoscalingInfo] = Field(
        None,
        description="Autoscaling configuration from HPA",
    )
    
    def to_dict(self, by_alias: bool = True) -> dict[str, Any]:
        """
        Convert to dictionary for YAML/JSON serialization.
        
        Args:
            by_alias: Use field aliases (camelCase) in output.
            
        Returns:
            Dictionary representation of the ModelSpec.
        """
        return self.model_dump(by_alias=by_alias, exclude_none=True)


class InferenceDeployment(BaseModel):
    """
    Internal representation of a discovered inference deployment.
    
    This model is used during the discovery phase before generating
    the final ModelSpec output.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    name: str = Field(..., description="Deployment name")
    namespace: str = Field(..., description="Kubernetes namespace")
    framework: str = Field(..., description="Detected framework (vllm, triton, tgi, unknown)")
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Detection confidence",
    )
    deployment_type: str = Field(..., description="Workload type")
    replicas: int = Field(1, description="Number of replicas")
    pod_names: list[str] = Field(default_factory=list, description="Associated pod names")
    detected_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Detection timestamp",
    )
    created_at: Optional[datetime] = Field(
        None,
        description="Earliest pod creation timestamp across this deployment's pods",
    )

    # Raw data for further processing
    container_image: Optional[str] = Field(None, description="Container image")
    container_args: list[str] = Field(default_factory=list, description="Container arguments")
    env_vars: dict[str, str] = Field(default_factory=dict, description="Environment variables")
    labels: dict[str, str] = Field(default_factory=dict, description="Pod labels")
    
    # Resource specifications
    gpu_count: int = Field(0, description="Number of GPUs requested")
    cpu_request: Optional[str] = Field(None, description="CPU request")
    memory_request: Optional[str] = Field(None, description="Memory request")
