"""
PIQC Schema models.

Pydantic models matching the piqc-scan.v0.1 JSON schema for
generating standardized facts bundles.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, ConfigDict


class Confidence(str, Enum):
    """Data confidence levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Severity(str, Enum):
    """Error severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class SourceType(str, Enum):
    """Source types for fact provenance."""
    SCANNER = "scanner"
    K8S_API = "k8s_api"
    POD_EXEC = "pod_exec"
    HTTP_METRICS = "http_metrics"
    PROMETHEUS = "prometheus"
    OTHER = "other"


class Source(BaseModel):
    """Provenance source for a fact."""

    model_config = ConfigDict(populate_by_name=True)

    type: SourceType = Field(..., description="Primary source category")
    method: Optional[str] = Field(None, description="Method detail (e.g., 'container_args')")
    ref: Optional[str] = Field(None, description="Reference identifier")


class FactValue(BaseModel):
    """A fact with provenance information."""

    model_config = ConfigDict(populate_by_name=True)

    value: Any = Field(..., description="Fact value (any JSON type)")
    source: Source = Field(..., description="How the fact was collected")
    data_confidence: Confidence = Field(
        ...,
        alias="dataConfidence",
        description="Confidence level",
    )
    observed_at: str = Field(
        ...,
        alias="observedAt",
        description="ISO 8601 timestamp",
    )
    units: Optional[str] = Field(None, description="Units (e.g., '%', 'GB')")
    staleness_sec: Optional[int] = Field(
        None,
        alias="stalenessSec",
        ge=0,
        description="Age of measurement in seconds",
    )


class ToolInfo(BaseModel):
    """Information about the scanning tool."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(..., min_length=1)
    version: str = Field(..., min_length=1)
    git_sha: Optional[str] = Field(None, alias="gitSha", min_length=6, max_length=64)


class ClusterInfo(BaseModel):
    """Kubernetes cluster information."""

    model_config = ConfigDict(populate_by_name=True)

    context: Optional[str] = None
    name: Optional[str] = None
    uid: Optional[str] = None


class ScopeInfo(BaseModel):
    """Scan scope information."""

    model_config = ConfigDict(populate_by_name=True)

    namespaces: Optional[list[str]] = Field(None, description="Namespaces scanned")
    selector: Optional[str] = Field(None, description="Label selector used")


class WorkloadObject(BaseModel):
    """A discovered workload with its facts."""

    model_config = ConfigDict(populate_by_name=True)

    workload_id: str = Field(
        ...,
        alias="workloadId",
        min_length=1,
        description="Stable ID (e.g., 'ns/<ns>/deploy/<name>')",
    )
    kind: str = Field(..., description="Kubernetes workload kind")
    name: str = Field(..., min_length=1)
    namespace: str = Field(..., min_length=1)
    images: Optional[list[str]] = Field(None, description="Container images")
    pods: Optional[list[str]] = Field(None, description="Associated pod names")
    facts: dict[str, FactValue] = Field(
        default_factory=dict,
        description="Fact map keyed by canonical fact names",
    )


class FactError(BaseModel):
    """Error collecting a fact."""

    model_config = ConfigDict(populate_by_name=True)

    workload_id: str = Field(..., alias="workloadId", min_length=1)
    fact_key: str = Field(..., alias="factKey", min_length=1)
    error: str = Field(..., min_length=1)
    severity: Severity
    source: Optional[Source] = None
    observed_at: Optional[str] = Field(None, alias="observedAt")


class PIQCBundle(BaseModel):
    """
    PIQC Scan Facts Bundle (v0.1).

    Root model for the piqc-facts.json output file.
    """

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = Field(
        "piqc-scan.v0.1",
        alias="schemaVersion",
        description="Schema version",
    )
    generated_at: str = Field(
        ...,
        alias="generatedAt",
        description="UTC timestamp of generation",
    )
    tool: ToolInfo
    cluster: Optional[ClusterInfo] = None
    scope: Optional[ScopeInfo] = None
    cluster_facts: Optional[dict[str, FactValue]] = Field(
        None,
        alias="clusterFacts",
        description="Cluster-level facts",
    )
    objects: list[WorkloadObject] = Field(
        ...,
        min_length=0,
        description="Discovered workloads",
    )
    fact_errors: list[FactError] = Field(
        default_factory=list,
        alias="factErrors",
        description="Non-fatal collection errors",
    )
    notes: Optional[list[str]] = Field(None, description="Informational notes")

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(by_alias=True, exclude_none=True)


# =============================================================================
# Facts Registry (Required and Optional Facts)
# =============================================================================

REQUIRED_FACTS = [
    "runtime.engineType",
    "vllm.dtype",
    "vllm.maxModelLen",
    "vllm.maxNumSeqs",
    "vllm.tensorParallelSize",
    "hardware.gpuType",
    "hardware.gpuMemoryGB",
    "hardware.gpuCount",
    "obs.gpu.utilAvgPct",
    "obs.gpu.memUtilAvgPct",
]

OPTIONAL_FACTS = [
    "runtime.engineVersion",
    "obs.vllm.tokensPerSec",
    # Extended facts beyond registry
    "model.name",
    "model.architecture",
    "model.parameters",
    "model.quantization",
    "vllm.gpuMemoryUtilization",
    "vllm.pipelineParallelSize",
    "vllm.enablePrefixCaching",
    "k8s.replicas",
    "k8s.cpuRequest",
    "k8s.memoryRequest",
    "obs.vllm.requestsRunning",
    "obs.vllm.requestsWaiting",
    "obs.vllm.kvCacheUsagePct",
    "obs.vllm.prefixCacheHitRate",
    "obs.gpu.temperatureC",
    "obs.gpu.powerDrawW",
    "endpoint.httpPort",
    # ==========================================================================
    # MVP Required Facts (added for compatibility)
    # ==========================================================================
    # Latency and throughput metrics
    "obs.tpot.p95Ms",
    "obs.requestLatency.p95Ms",
    "obs.requestLatency.p99Ms",
    "obs.tps.avg",
    "obs.rps.avg",
    "obs.effectiveBatch.p95",
    # Alias facts for backwards compatibility
    "model.maxModelLen",
    "runtime.dtype",
    "runtime.tensorParallel",
    "model.parameterCount",
    # Hardware capability facts
    "hardware.supportsFastBf16",
    "hardware.interconnect",
    # Autoscaling facts
    "autoscaling.enabled",
    "autoscaling.metricType",
]

