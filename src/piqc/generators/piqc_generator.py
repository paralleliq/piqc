"""
PIQC Facts Generator.

Transforms ModelSpec data into PIQC facts bundle format (piqc-scan.v0.1)
with full provenance tracking and error reporting.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from piqc import __version__
from piqc.collectors.vllm_collector import derive_parallelism_strategy
from piqc.core.orchestrator import (
    FragmentedNodeInfo,
    PendingGPUPod,
    UnallocatedNodeInfo,
    WorkloadPlacement,
)
from piqc.models.modelspec import ModelSpec
from piqc.models.piqc_schema import (
    Confidence,
    ClusterInfo,
    FactError,
    FactValue,
    PIQCBundle,
    REQUIRED_FACTS,
    ScopeInfo,
    Severity,
    Source,
    SourceType,
    ToolInfo,
    WorkloadObject,
)
from piqc.utils.logger import get_logger


logger = get_logger(__name__)


class PIQCGenerator:
    """
    Generates PIQC facts bundle from ModelSpec data.

    Extracts facts with provenance tracking and reports missing
    required facts as factErrors.
    """

    def __init__(self) -> None:
        """Initialize the generator."""
        self._fact_errors: list[FactError] = []
        self._timestamp: str = ""

    def generate(
        self,
        modelspecs: list[ModelSpec],
        output_path: str,
        cluster_context: Optional[str] = None,
        cluster_name: Optional[str] = None,
        namespaces: Optional[list[str]] = None,
        unallocated_nodes: Optional[list[UnallocatedNodeInfo]] = None,
        fragmented_nodes: Optional[list[FragmentedNodeInfo]] = None,
        pending_gpu_pods: Optional[list[PendingGPUPod]] = None,
        workload_placements: Optional[dict[str, WorkloadPlacement]] = None,
    ) -> str:
        """
        Generate piqc-facts.json file.

        Args:
            modelspecs: List of ModelSpec objects.
            output_path: Output directory path.
            cluster_context: K8s context name.
            cluster_name: K8s cluster name.
            namespaces: Scanned namespaces.
            unallocated_nodes: Nodes with GPU capacity no pod has requested
                (from Orchestrator._analyze_node_capacity). Each becomes its
                own node-scoped WorkloadObject, since piqc's bundle format
                has no separate node-level object type today.
            fragmented_nodes: Nodes with leftover GPU slots too small for any
                known model (from Orchestrator._analyze_fragmented_nodes).
                Already computed for the CLI's own table report
                (table_generator.py) but never reached this bundle before --
                folded into each pending pod's own WorkloadObject alongside
                unallocated_nodes (see _build_pending_pod_workload), not
                emitted as a separate object: fragmentation_v1's `when`
                clause evaluates one workload's fact store at a time, so a
                standalone cluster-scoped object's facts would never be
                visible to the same evaluation as a pending job's facts.
            pending_gpu_pods: Pods stuck Pending because no node can satisfy
                their GPU request (from Orchestrator._analyze_pending_gpu_pods).
                Same "computed for the CLI report, never wired here" gap as
                fragmented_nodes -- each becomes its own pod-scoped
                WorkloadObject, carrying both its own demand facts and the
                cluster-wide fragmentation facts above.
            workload_placements: Per-deployment node spread and node-pool
                GPU capacity (from Orchestrator._analyze_workload_placement),
                keyed by "namespace/name" -- InferenceDeployment objects
                (which carry the pod_names this is computed from) don't
                survive past ModelSpec conversion, so this is the join key
                back to each spec. Attached directly onto the matching
                ModelSpec's own WorkloadObject (see _extract_placement_facts),
                not a synthetic object -- these facts genuinely belong to
                that workload, unlike fragmentation's cluster-wide numbers.

        Returns:
            Path to generated file.
        """
        self._timestamp = datetime.utcnow().isoformat() + "Z"
        self._fact_errors = []
        workload_placements = workload_placements or {}

        # Build workload objects
        objects = [
            self._build_workload(spec, workload_placements) for spec in modelspecs
        ]
        objects.extend(self._build_node_workload(node) for node in (unallocated_nodes or []))
        objects.extend(
            self._build_pending_pod_workload(pod, unallocated_nodes or [], fragmented_nodes or [])
            for pod in (pending_gpu_pods or [])
        )

        # Build bundle
        bundle = PIQCBundle(
            schema_version="piqc-scan.v0.1",
            generated_at=self._timestamp,
            tool=ToolInfo(
                name="piqc",
                version=__version__,
            ),
            cluster=ClusterInfo(
                context=cluster_context,
                name=cluster_name,
            ) if cluster_context or cluster_name else None,
            scope=ScopeInfo(
                namespaces=namespaces,
            ) if namespaces else None,
            objects=objects,
            fact_errors=self._fact_errors,
            notes=self._generate_notes(modelspecs),
        )

        # Write output
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "piqc-facts.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(bundle.to_dict(), f, indent=2)

        logger.info(f"Generated PIQC facts bundle: {output_file}")
        return str(output_file)

    def _build_node_workload(self, node: UnallocatedNodeInfo) -> WorkloadObject:
        """Build a node-scoped WorkloadObject from real node GPU allocation data.

        Deliberately skips _check_required_facts — those checks assume a
        vLLM/runtime workload, which a node-level object isn't.
        """
        workload_id = f"ns/gpu-pool/node/{node.node_name}"

        facts: dict[str, FactValue] = {
            "hardware.gpuType": FactValue(
                value=node.gpu_type,
                source=Source(type=SourceType.K8S_API, method="node_status"),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            ),
            "hardware.gpuCount": FactValue(
                value=node.total_gpus,
                source=Source(type=SourceType.K8S_API, method="node_status"),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            ),
            "node.allocatedGpuCount": FactValue(
                value=node.allocated_gpus,
                source=Source(type=SourceType.K8S_API, method="pod_spec"),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            ),
            "node.unallocatedGpuCount": FactValue(
                value=node.unallocated_gpus,
                source=Source(type=SourceType.K8S_API, method="node_status"),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            ),
        }

        return WorkloadObject(
            workload_id=workload_id,
            kind="Node",
            name=node.node_name,
            namespace="gpu-pool",
            images=None,
            pods=None,
            facts=facts,
        )

    def _cluster_fragmentation_facts(
        self,
        unallocated_nodes: list[UnallocatedNodeInfo],
        fragmented_nodes: list[FragmentedNodeInfo],
    ) -> dict[str, FactValue]:
        """Cluster-wide fragmentation facts fragmentation_v1 needs, from data
        _analyze_node_capacity()/_analyze_fragmented_nodes() already compute
        today for the CLI's own table report -- this was never emitted into
        the facts bundle before.

        Folded directly into each pending pod's own WorkloadObject (see
        _build_pending_pod_workload) rather than emitted as its own
        cluster-scoped object: fragmentation_v1's `when` clause evaluates one
        workload's fact store at a time (_evaluate_from_rules in
        advisor_integration.py takes a single flat `scan.facts` dict, not a
        join across objects), so a standalone object's facts would never be
        visible to the same evaluation as a pending job's demand facts --
        confirmed by testing a real /v1/ingest payload with the two split
        across separate objects: the rule never fired.

        unallocated_nodes is the exhaustive list of every node with any free
        GPU slot (from a fully-empty node down to a single stray GPU);
        fragmented_nodes is the subset whose leftover slots are too small to
        fit any known model. GPUs can't be combined across nodes, so the
        largest single node's free count is the real usable ceiling, not the
        cluster-wide sum -- same reasoning FragmentedNodeInfo's own docstring
        gives for why a fragmented node's slots are "stranded."
        """
        total_free = sum(n.unallocated_gpus for n in unallocated_nodes)
        largest_block = max((n.unallocated_gpus for n in unallocated_nodes), default=0)

        return {
            "cluster.totalFreeGpus": FactValue(
                value=total_free,
                source=Source(type=SourceType.K8S_API, method="node_status"),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            ),
            "cluster.largestContiguousBlock": FactValue(
                value=largest_block,
                source=Source(type=SourceType.K8S_API, method="node_status"),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            ),
            "cluster.fragmentedNodeCount": FactValue(
                value=len(fragmented_nodes),
                source=Source(type=SourceType.K8S_API, method="node_status"),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            ),
        }

    def _build_pending_pod_workload(
        self,
        pod: PendingGPUPod,
        unallocated_nodes: list[UnallocatedNodeInfo],
        fragmented_nodes: list[FragmentedNodeInfo],
    ) -> WorkloadObject:
        """Build a pod-scoped WorkloadObject for a pod stuck Pending on GPU
        capacity, from _analyze_pending_gpu_pods() -- same "computed for the
        CLI report, never wired into the facts bundle" gap as the cluster
        fragmentation facts folded in here.

        deployment.gpuType is deliberately not set here: unlike a running
        pod (whose actual GPU model piqc reads live from the container via
        gpu_collector.py), a Pending pod was never scheduled to a node, so
        there's no live GPU to query and no reliable static source for the
        model either -- resource requests only say "how many," not "which
        kind." Left absent rather than guessed; fragmentation_v1's `when`
        clause doesn't require it, only the messageTemplate references it.
        """
        workload_id = f"ns/{pod.namespace}/pod/{pod.pod_name}"

        facts: dict[str, FactValue] = {
            "job.pendingGpuCount": FactValue(
                value=pod.gpus_needed,
                source=Source(type=SourceType.K8S_API, method="pod_spec"),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            ),
            "job.pendingSinceMinutes": FactValue(
                value=round(pod.pending_minutes, 1),
                source=Source(type=SourceType.K8S_API, method="pod_metadata"),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            ),
        }
        facts.update(self._cluster_fragmentation_facts(unallocated_nodes, fragmented_nodes))
        if pod.parallelism_strategy is not None:
            facts["deployment.parallelismStrategy"] = FactValue(
                value=pod.parallelism_strategy,
                source=Source(type=SourceType.K8S_API, method="container_args"),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            )

        return WorkloadObject(
            workload_id=workload_id,
            kind="Pod",
            name=pod.pod_name,
            namespace=pod.namespace,
            images=None,
            pods=None,
            facts=facts,
        )

    def _build_workload(
        self, spec: ModelSpec, workload_placements: dict[str, WorkloadPlacement]
    ) -> WorkloadObject:
        """Build a WorkloadObject from a ModelSpec."""
        workload_id = f"ns/{spec.metadata.namespace}/{spec.kubernetes.deployment_type.lower()}/{spec.metadata.name}"

        facts: dict[str, FactValue] = {}

        # Extract all facts
        self._extract_runtime_facts(spec, facts, workload_id)
        self._extract_vllm_facts(spec, facts, workload_id)
        self._extract_hardware_facts(spec, facts, workload_id)
        self._extract_observed_facts(spec, facts, workload_id)
        self._extract_model_facts(spec, facts, workload_id)
        self._extract_k8s_facts(spec, facts, workload_id)
        self._extract_endpoint_facts(spec, facts, workload_id)
        self._extract_autoscaling_facts(spec, facts, workload_id)
        self._extract_placement_facts(spec, facts, workload_placements)

        # Check for missing required facts
        self._check_required_facts(facts, workload_id)

        return WorkloadObject(
            workload_id=workload_id,
            kind=spec.kubernetes.deployment_type,
            name=spec.metadata.name,
            namespace=spec.metadata.namespace,
            images=[spec.kubernetes.image] if spec.kubernetes.image else None,
            pods=None,  # Could extract from GPUInfo.pod_name
            facts=facts,
        )

    def _extract_runtime_facts(
        self,
        spec: ModelSpec,
        facts: dict[str, FactValue],
        workload_id: str,
    ) -> None:
        """Extract runtime identification facts."""
        # runtime.engineType (required)
        if spec.engine.name:
            facts["runtime.engineType"] = FactValue(
                value=spec.engine.name,
                source=Source(
                    type=SourceType.SCANNER,
                    method="framework_detection",
                ),
                data_confidence=self._confidence_from_float(spec.engine.detection_confidence),
                observed_at=self._timestamp,
            )

        # runtime.engineVersion (optional)
        if spec.kubernetes.image_tag:
            facts["runtime.engineVersion"] = FactValue(
                value=spec.kubernetes.image_tag,
                source=Source(
                    type=SourceType.K8S_API,
                    method="container_image_tag",
                ),
                data_confidence=Confidence.MEDIUM,
                observed_at=self._timestamp,
            )

        # ---------------------------------------------------------------
        # Engine-agnostic aliases, read from spec.inference directly.
        # spec.inference (InferenceConfig) is already generically shaped,
        # not vLLM-specific, so any future engine parser that populates it
        # gets these facts for free. Do not move this logic into an
        # engine-gated extractor (see piqc#8 for the SGLang follow-up).
        # ---------------------------------------------------------------
        inference = spec.inference

        # model.quantization
        if inference.quantization:
            facts["model.quantization"] = FactValue(
                value=inference.quantization,
                source=Source(
                    type=SourceType.K8S_API,
                    method="container_args",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            )

        # model.maxModelLen (alias of the engine-specific max context length fact)
        if inference.max_model_len:
            facts["model.maxModelLen"] = FactValue(
                value=inference.max_model_len,
                source=Source(
                    type=SourceType.K8S_API,
                    method="container_args",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
                units="tokens",
            )

        # runtime.dtype (alias of the engine-specific precision fact)
        if inference.precision:
            facts["runtime.dtype"] = FactValue(
                value=inference.precision,
                source=Source(
                    type=SourceType.K8S_API,
                    method="container_args",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            )

        # runtime.tensorParallel (alias of the engine-specific tensor parallel size fact)
        if inference.tensor_parallel_size:
            facts["runtime.tensorParallel"] = FactValue(
                value=inference.tensor_parallel_size,
                source=Source(
                    type=SourceType.K8S_API,
                    method="container_args",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            )

        # deployment.parallelismStrategy -- tensor_parallel_cross_node_v1 and
        # fragmentation_v1 both gate on this categorical value directly; it's
        # a pure derivation from the two sizes above, engine-agnostic like
        # them, so it belongs here rather than in _extract_vllm_facts.
        strategy = derive_parallelism_strategy(
            inference.tensor_parallel_size, inference.pipeline_parallel_size
        )
        if strategy is not None:
            facts["deployment.parallelismStrategy"] = FactValue(
                value=strategy,
                source=Source(
                    type=SourceType.K8S_API,
                    method="container_args",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            )

    def _extract_vllm_facts(
        self,
        spec: ModelSpec,
        facts: dict[str, FactValue],
        workload_id: str,
    ) -> None:
        """Extract vLLM configuration facts."""
        if spec.engine.name != "vllm":
            return

        inference = spec.inference

        # vllm.dtype (required)
        if inference.precision:
            facts["vllm.dtype"] = FactValue(
                value=inference.precision,
                source=Source(
                    type=SourceType.K8S_API,
                    method="container_args",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            )

        # vllm.maxModelLen (required)
        if inference.max_model_len:
            facts["vllm.maxModelLen"] = FactValue(
                value=inference.max_model_len,
                source=Source(
                    type=SourceType.K8S_API,
                    method="container_args",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
                units="tokens",
            )

        # vllm.maxNumSeqs (required)
        if inference.max_sequences:
            facts["vllm.maxNumSeqs"] = FactValue(
                value=inference.max_sequences,
                source=Source(
                    type=SourceType.K8S_API,
                    method="container_args",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            )

        # vllm.tensorParallelSize (required)
        if inference.tensor_parallel_size:
            facts["vllm.tensorParallelSize"] = FactValue(
                value=inference.tensor_parallel_size,
                source=Source(
                    type=SourceType.K8S_API,
                    method="container_args",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            )

        # vllm.pipelineParallelSize (optional extended)
        if inference.pipeline_parallel_size:
            facts["vllm.pipelineParallelSize"] = FactValue(
                value=inference.pipeline_parallel_size,
                source=Source(
                    type=SourceType.K8S_API,
                    method="container_args",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            )

        # vllm.gpuMemoryUtilization (optional extended)
        if spec.resources.gpu_memory_utilization:
            facts["vllm.gpuMemoryUtilization"] = FactValue(
                value=spec.resources.gpu_memory_utilization,
                source=Source(
                    type=SourceType.K8S_API,
                    method="container_args",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
                units="%",
            )

        # Note: model.quantization, model.maxModelLen, runtime.dtype, and
        # runtime.tensorParallel used to be emitted here as vLLM-only aliases.
        # They now live in _extract_runtime_facts(), reading from spec.inference
        # directly, so they populate for any engine, not just vLLM.

    def _extract_hardware_facts(
        self,
        spec: ModelSpec,
        facts: dict[str, FactValue],
        workload_id: str,
    ) -> None:
        """Extract hardware-related facts."""
        gpus = spec.resources.gpus

        # hardware.gpuCount (required)
        gpu_count = len(gpus) if gpus else 0
        facts["hardware.gpuCount"] = FactValue(
            value=gpu_count,
            source=Source(
                type=SourceType.K8S_API,
                method="pod_resources",
            ),
            data_confidence=Confidence.HIGH,
            observed_at=self._timestamp,
        )

        if gpus and len(gpus) > 0:
            gpu = gpus[0]

            # hardware.gpuType (required)
            if gpu.type:
                facts["hardware.gpuType"] = FactValue(
                    value=gpu.type,
                    source=Source(
                        type=SourceType.POD_EXEC,
                        method="nvidia-smi",
                        ref=gpu.pod_name,
                    ),
                    data_confidence=Confidence.HIGH,
                    observed_at=self._timestamp,
                )

            # hardware.gpuMemoryGB (required)
            if gpu.memory_total:
                memory_gb = self._parse_memory_to_gb(gpu.memory_total)
                facts["hardware.gpuMemoryGB"] = FactValue(
                    value=memory_gb,
                    source=Source(
                        type=SourceType.POD_EXEC,
                        method="nvidia-smi",
                        ref=gpu.pod_name,
                    ),
                    data_confidence=Confidence.HIGH,
                    observed_at=self._timestamp,
                    units="GB",
                )

            # =================================================================
            # MVP Required Facts - Hardware Capabilities
            # =================================================================

            # hardware.supportsFastBf16 (derived from GPU type)
            if gpu.type:
                supports_bf16 = self._gpu_supports_fast_bf16(gpu.type)
                facts["hardware.supportsFastBf16"] = FactValue(
                    value=supports_bf16,
                    source=Source(
                        type=SourceType.SCANNER,
                        method="gpu_capability_inference",
                    ),
                    data_confidence=Confidence.MEDIUM,
                    observed_at=self._timestamp,
                )

            # hardware.interconnect (inferred from TP size and GPU type)
            if gpu.type and gpu_count > 1:
                interconnect = self._infer_gpu_interconnect(gpu.type, gpu_count)
                facts["hardware.interconnect"] = FactValue(
                    value=interconnect,
                    source=Source(
                        type=SourceType.SCANNER,
                        method="gpu_topology_inference",
                    ),
                    data_confidence=Confidence.LOW,  # Inferred, not measured
                    observed_at=self._timestamp,
                )

    def _extract_observed_facts(
        self,
        spec: ModelSpec,
        facts: dict[str, FactValue],
        workload_id: str,
    ) -> None:
        """Extract observed runtime metrics."""
        gpus = spec.resources.gpus

        if gpus and len(gpus) > 0:
            gpu = gpus[0]

            # obs.gpu.utilAvgPct (required)
            if gpu.utilization is not None:
                facts["obs.gpu.utilAvgPct"] = FactValue(
                    value=gpu.utilization,
                    source=Source(
                        type=SourceType.POD_EXEC,
                        method="nvidia-smi",
                        ref=gpu.pod_name,
                    ),
                    data_confidence=Confidence.MEDIUM,
                    observed_at=self._timestamp,
                    units="%",
                )

            # obs.gpu.memUtilAvgPct (required)
            if gpu.memory_total and gpu.memory_used:
                total_mb = self._parse_memory_to_mb(gpu.memory_total)
                used_mb = self._parse_memory_to_mb(gpu.memory_used)
                if total_mb > 0:
                    mem_pct = round((used_mb / total_mb) * 100, 1)
                    facts["obs.gpu.memUtilAvgPct"] = FactValue(
                        value=mem_pct,
                        source=Source(
                            type=SourceType.POD_EXEC,
                            method="nvidia-smi",
                            ref=gpu.pod_name,
                        ),
                        data_confidence=Confidence.MEDIUM,
                        observed_at=self._timestamp,
                        units="%",
                    )

            # obs.gpu.temperatureC (optional extended)
            if gpu.temperature is not None:
                facts["obs.gpu.temperatureC"] = FactValue(
                    value=gpu.temperature,
                    source=Source(
                        type=SourceType.POD_EXEC,
                        method="nvidia-smi",
                    ),
                    data_confidence=Confidence.HIGH,
                    observed_at=self._timestamp,
                    units="C",
                )

            # obs.gpu.powerDrawW (optional extended)
            if gpu.power_draw is not None:
                facts["obs.gpu.powerDrawW"] = FactValue(
                    value=gpu.power_draw,
                    source=Source(
                        type=SourceType.POD_EXEC,
                        method="nvidia-smi",
                    ),
                    data_confidence=Confidence.HIGH,
                    observed_at=self._timestamp,
                    units="W",
                )

        cpu = spec.resources.cpu
        if cpu is not None:
            # obs.cpu.utilAvgPct (optional extended)
            facts["obs.cpu.utilAvgPct"] = FactValue(
                value=cpu.utilization_percent,
                source=Source(
                    type=SourceType.POD_EXEC,
                    method="proc_stat",
                    ref=cpu.pod_name,
                ),
                data_confidence=Confidence.MEDIUM,
                observed_at=self._timestamp,
                units="%",
            )

            # obs.cpu.iowaitAvgPct (optional extended)
            facts["obs.cpu.iowaitAvgPct"] = FactValue(
                value=cpu.iowait_percent,
                source=Source(
                    type=SourceType.POD_EXEC,
                    method="proc_stat",
                    ref=cpu.pod_name,
                ),
                data_confidence=Confidence.MEDIUM,
                observed_at=self._timestamp,
                units="%",
            )

        # vLLM runtime metrics
        if spec.runtime_state and spec.runtime_state.vllm:
            vllm = spec.runtime_state.vllm

            # obs.vllm.tokensPerSec (optional)
            if vllm.generation_tokens_per_sec:
                facts["obs.vllm.tokensPerSec"] = FactValue(
                    value=vllm.generation_tokens_per_sec,
                    source=Source(
                        type=SourceType.HTTP_METRICS,
                        method="GET /metrics",
                    ),
                    data_confidence=Confidence.MEDIUM,
                    observed_at=vllm.collection_timestamp or self._timestamp,
                    units="tokens/s",
                )

            # obs.vllm.requestsRunning (extended)
            if vllm.requests_running is not None:
                facts["obs.vllm.requestsRunning"] = FactValue(
                    value=vllm.requests_running,
                    source=Source(
                        type=SourceType.HTTP_METRICS,
                        method="GET /metrics",
                    ),
                    data_confidence=Confidence.HIGH,
                    observed_at=vllm.collection_timestamp or self._timestamp,
                )

            # obs.vllm.requestsWaiting (extended)
            if vllm.requests_waiting is not None:
                facts["obs.vllm.requestsWaiting"] = FactValue(
                    value=vllm.requests_waiting,
                    source=Source(
                        type=SourceType.HTTP_METRICS,
                        method="GET /metrics",
                    ),
                    data_confidence=Confidence.HIGH,
                    observed_at=vllm.collection_timestamp or self._timestamp,
                )

            # obs.vllm.kvCacheUsagePct (extended)
            if vllm.gpu_cache_usage_percent:
                facts["obs.vllm.kvCacheUsagePct"] = FactValue(
                    value=vllm.gpu_cache_usage_percent,
                    source=Source(
                        type=SourceType.HTTP_METRICS,
                        method="GET /metrics",
                    ),
                    data_confidence=Confidence.HIGH,
                    observed_at=vllm.collection_timestamp or self._timestamp,
                    units="%",
                )

            # obs.vllm.prefixCacheHitRate (extended)
            if vllm.prefix_cache_hit_rate:
                facts["obs.vllm.prefixCacheHitRate"] = FactValue(
                    value=vllm.prefix_cache_hit_rate,
                    source=Source(
                        type=SourceType.HTTP_METRICS,
                        method="GET /metrics",
                    ),
                    data_confidence=Confidence.HIGH,
                    observed_at=vllm.collection_timestamp or self._timestamp,
                )

            # =================================================================
            # MVP Required Facts - Latency and Throughput Metrics
            # =================================================================

            # obs.tpot.p95Ms (Time Per Output Token p95 in milliseconds)
            if vllm.tpot_p95 is not None:
                facts["obs.tpot.p95Ms"] = FactValue(
                    value=round(vllm.tpot_p95 * 1000, 2),  # Convert seconds to ms
                    source=Source(
                        type=SourceType.HTTP_METRICS,
                        method="GET /metrics",
                    ),
                    data_confidence=Confidence.MEDIUM,
                    observed_at=vllm.collection_timestamp or self._timestamp,
                    units="ms",
                )

            # obs.requestLatency.p95Ms (End-to-end latency p95 in milliseconds)
            if vllm.e2e_p95 is not None:
                facts["obs.requestLatency.p95Ms"] = FactValue(
                    value=round(vllm.e2e_p95 * 1000, 2),  # Convert seconds to ms
                    source=Source(
                        type=SourceType.HTTP_METRICS,
                        method="GET /metrics",
                    ),
                    data_confidence=Confidence.MEDIUM,
                    observed_at=vllm.collection_timestamp or self._timestamp,
                    units="ms",
                )

            # obs.requestLatency.p99Ms (End-to-end latency p99 in milliseconds)
            if vllm.e2e_p99 is not None:
                facts["obs.requestLatency.p99Ms"] = FactValue(
                    value=round(vllm.e2e_p99 * 1000, 2),  # Convert seconds to ms
                    source=Source(
                        type=SourceType.HTTP_METRICS,
                        method="GET /metrics",
                    ),
                    data_confidence=Confidence.MEDIUM,
                    observed_at=vllm.collection_timestamp or self._timestamp,
                    units="ms",
                )

            # obs.tps.avg (Average Tokens Per Second throughput)
            if vllm.generation_tokens_per_sec is not None:
                facts["obs.tps.avg"] = FactValue(
                    value=vllm.generation_tokens_per_sec,
                    source=Source(
                        type=SourceType.HTTP_METRICS,
                        method="GET /metrics",
                    ),
                    data_confidence=Confidence.MEDIUM,
                    observed_at=vllm.collection_timestamp or self._timestamp,
                    units="tokens/s",
                )

            # obs.effectiveBatch.p95 (Effective batch size - use running requests as proxy)
            if vllm.requests_running is not None and vllm.requests_running > 0:
                facts["obs.effectiveBatch.p95"] = FactValue(
                    value=vllm.requests_running,
                    source=Source(
                        type=SourceType.HTTP_METRICS,
                        method="GET /metrics",
                    ),
                    data_confidence=Confidence.LOW,  # Using running as proxy for batch
                    observed_at=vllm.collection_timestamp or self._timestamp,
                )

            # obs.rps.avg (Requests Per Second - using active requests as instantaneous proxy)
            # Note: True RPS requires rate calculation from counter, using running+waiting as proxy
            if vllm.requests_running is not None or vllm.requests_waiting is not None:
                active_requests = (vllm.requests_running or 0) + (vllm.requests_waiting or 0)
                if active_requests > 0:
                    facts["obs.rps.avg"] = FactValue(
                        value=active_requests,
                        source=Source(
                            type=SourceType.HTTP_METRICS,
                            method="GET /metrics",
                        ),
                        data_confidence=Confidence.LOW,  # Proxy for RPS
                        observed_at=vllm.collection_timestamp or self._timestamp,
                        units="requests",
                    )

    def _extract_model_facts(
        self,
        spec: ModelSpec,
        facts: dict[str, FactValue],
        workload_id: str,
    ) -> None:
        """Extract model information facts (extended)."""
        model = spec.model

        if model.name:
            facts["model.name"] = FactValue(
                value=model.name,
                source=Source(
                    type=SourceType.K8S_API,
                    method=model.identification_method or "unknown",
                ),
                data_confidence=self._confidence_from_float(model.confidence),
                observed_at=self._timestamp,
            )

        if model.architecture:
            facts["model.architecture"] = FactValue(
                value=model.architecture,
                source=Source(
                    type=SourceType.SCANNER,
                    method="model_name_inference",
                ),
                data_confidence=Confidence.MEDIUM,
                observed_at=self._timestamp,
            )

        if model.parameters:
            facts["model.parameters"] = FactValue(
                value=model.parameters,
                source=Source(
                    type=SourceType.SCANNER,
                    method="model_name_inference",
                ),
                data_confidence=Confidence.MEDIUM,
                observed_at=self._timestamp,
            )

            # model.parameterCount (MVP - numeric value parsed from "7B" format)
            param_count = self._parse_parameter_count(model.parameters)
            if param_count is not None:
                facts["model.parameterCount"] = FactValue(
                    value=param_count,
                    source=Source(
                        type=SourceType.SCANNER,
                        method="model_name_inference",
                    ),
                    data_confidence=Confidence.MEDIUM,
                    observed_at=self._timestamp,
                )

    def _extract_k8s_facts(
        self,
        spec: ModelSpec,
        facts: dict[str, FactValue],
        workload_id: str,
    ) -> None:
        """Extract Kubernetes metadata facts (extended)."""
        resources = spec.resources

        # k8s.replicas
        facts["k8s.replicas"] = FactValue(
            value=resources.replicas,
            source=Source(
                type=SourceType.K8S_API,
                method="deployment_spec",
            ),
            data_confidence=Confidence.HIGH,
            observed_at=self._timestamp,
        )

        # k8s.cpuRequest
        if resources.cpu_request:
            facts["k8s.cpuRequest"] = FactValue(
                value=resources.cpu_request,
                source=Source(
                    type=SourceType.K8S_API,
                    method="pod_spec",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            )

        # k8s.memoryRequest
        if resources.memory_request:
            facts["k8s.memoryRequest"] = FactValue(
                value=resources.memory_request,
                source=Source(
                    type=SourceType.K8S_API,
                    method="pod_spec",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            )

        # k8s.ageHours — workload age, derived from pod creation timestamp.
        # Already collected for the CLI "Age" column; surfaced as a fact so
        # idle/forgotten-workload detection can use it too.
        creation_timestamp = spec.kubernetes.creation_timestamp
        if creation_timestamp:
            try:
                created = datetime.fromisoformat(creation_timestamp)
            except ValueError:
                created = None
            if created is not None:
                now = (
                    datetime.now(created.tzinfo)
                    if created.tzinfo
                    else datetime.now(timezone.utc).replace(tzinfo=None)
                )
                age_hours = (now - created).total_seconds() / 3600
                facts["k8s.ageHours"] = FactValue(
                    value=round(age_hours, 2),
                    source=Source(
                        type=SourceType.K8S_API,
                        method="pod_spec",
                    ),
                    data_confidence=Confidence.HIGH,
                    observed_at=self._timestamp,
                    units="hours",
                )

    def _extract_endpoint_facts(
        self,
        spec: ModelSpec,
        facts: dict[str, FactValue],
        workload_id: str,
    ) -> None:
        """Extract endpoint facts (extended)."""
        if spec.endpoints:
            for endpoint in spec.endpoints:
                if endpoint.type == "http":
                    facts["endpoint.httpPort"] = FactValue(
                        value=endpoint.port,
                        source=Source(
                            type=SourceType.K8S_API,
                            method="container_ports",
                        ),
                        data_confidence=Confidence.HIGH,
                        observed_at=self._timestamp,
                    )
                    break

    def _extract_autoscaling_facts(
        self,
        spec: ModelSpec,
        facts: dict[str, FactValue],
        workload_id: str,
    ) -> None:
        """Extract autoscaling facts from HPA configuration."""
        if spec.autoscaling:
            autoscaling = spec.autoscaling

            # autoscaling.enabled
            facts["autoscaling.enabled"] = FactValue(
                value=autoscaling.enabled,
                source=Source(
                    type=SourceType.K8S_API,
                    method="hpa_spec",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            )

            # autoscaling.metricType
            if autoscaling.metric_type:
                facts["autoscaling.metricType"] = FactValue(
                    value=autoscaling.metric_type,
                    source=Source(
                        type=SourceType.K8S_API,
                        method="hpa_spec",
                    ),
                    data_confidence=Confidence.HIGH,
                    observed_at=self._timestamp,
                )
        else:
            # No HPA configured - autoscaling disabled
            facts["autoscaling.enabled"] = FactValue(
                value=False,
                source=Source(
                    type=SourceType.SCANNER,
                    method="hpa_absence",
                ),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            )

    def _extract_placement_facts(
        self,
        spec: ModelSpec,
        facts: dict[str, FactValue],
        workload_placements: dict[str, WorkloadPlacement],
    ) -> None:
        """placement.nodeCount / node.gpuCount -- tensor_parallel_cross_node_v1's
        two facts, from Orchestrator._analyze_workload_placement(). Looked
        up by "namespace/name" since the WorkloadPlacement was computed
        against the InferenceDeployment this ModelSpec came from, not the
        spec itself (see that method's own docstring for why).

        Silently omitted (not zeroed) when no placement data exists for
        this workload -- e.g. every pod's node couldn't be resolved, or
        placement analysis wasn't run at all -- same "absent, not guessed"
        convention _build_pending_pod_workload uses for deployment.gpuType.
        """
        placement = workload_placements.get(f"{spec.metadata.namespace}/{spec.metadata.name}")
        if placement is None:
            return

        facts["placement.nodeCount"] = FactValue(
            value=placement.node_count,
            source=Source(type=SourceType.K8S_API, method="pod_spec"),
            data_confidence=Confidence.HIGH,
            observed_at=self._timestamp,
        )
        if placement.node_gpu_count is not None:
            facts["node.gpuCount"] = FactValue(
                value=placement.node_gpu_count,
                source=Source(type=SourceType.K8S_API, method="node_status"),
                data_confidence=Confidence.HIGH,
                observed_at=self._timestamp,
            )

    def _check_required_facts(
        self,
        facts: dict[str, FactValue],
        workload_id: str,
    ) -> None:
        """Check for missing required facts and add to factErrors."""
        for fact_key in REQUIRED_FACTS:
            if fact_key not in facts:
                self._fact_errors.append(FactError(
                    workload_id=workload_id,
                    fact_key=fact_key,
                    error=f"Required fact '{fact_key}' could not be collected",
                    severity=Severity.WARNING,
                    source=Source(type=SourceType.SCANNER, method="fact_validation"),
                    observed_at=self._timestamp,
                ))

    def _confidence_from_float(self, value: float) -> Confidence:
        """Convert float confidence to enum."""
        if value >= 0.8:
            return Confidence.HIGH
        elif value >= 0.5:
            return Confidence.MEDIUM
        return Confidence.LOW

    def _parse_memory_to_gb(self, memory_str: str) -> float:
        """Parse memory string to GB."""
        memory_str = memory_str.strip().upper()
        if memory_str.endswith("GB"):
            return float(memory_str[:-2])
        elif memory_str.endswith("MB"):
            return round(float(memory_str[:-2]) / 1024, 2)
        elif memory_str.endswith("TB"):
            return float(memory_str[:-2]) * 1024
        return 0.0

    def _parse_memory_to_mb(self, memory_str: str) -> float:
        """Parse memory string to MB."""
        memory_str = memory_str.strip().upper()
        if memory_str.endswith("GB"):
            return float(memory_str[:-2]) * 1024
        elif memory_str.endswith("MB"):
            return float(memory_str[:-2])
        elif memory_str.endswith("TB"):
            return float(memory_str[:-2]) * 1024 * 1024
        return 0.0

    def _parse_parameter_count(self, param_str: str) -> Optional[int]:
        """
        Parse parameter count string to numeric value.
        
        Examples:
            '7B' -> 7000000000
            '13B' -> 13000000000
            '70B' -> 70000000000
            '1.5B' -> 1500000000
            '8x7B' -> 56000000000 (MoE models)
        """
        import re
        
        if not param_str:
            return None
        
        param_str = param_str.strip().upper()
        
        # Handle MoE format like "8x7B"
        moe_match = re.match(r'(\d+)X(\d+\.?\d*)B', param_str)
        if moe_match:
            experts = int(moe_match.group(1))
            per_expert = float(moe_match.group(2))
            return int(experts * per_expert * 1_000_000_000)
        
        # Handle standard format like "7B", "70B", "1.5B"
        std_match = re.match(r'(\d+\.?\d*)B', param_str)
        if std_match:
            billions = float(std_match.group(1))
            return int(billions * 1_000_000_000)
        
        # Handle millions format "405M"
        m_match = re.match(r'(\d+\.?\d*)M', param_str)
        if m_match:
            millions = float(m_match.group(1))
            return int(millions * 1_000_000)
        
        return None

    def _gpu_supports_fast_bf16(self, gpu_type: str) -> bool:
        """
        Determine if GPU supports fast BF16 operations.
        
        Fast BF16 is available on:
        - NVIDIA Ampere (A100, A10, A30, A40, A6000)
        - NVIDIA Hopper (H100, H200)
        - NVIDIA Ada Lovelace (L40, L40S, RTX 4090)
        - NVIDIA Blackwell (B100, B200)
        
        Returns:
            True if GPU supports fast BF16, False otherwise.
        """
        gpu_upper = gpu_type.upper()
        
        # Ampere GPUs (compute capability 8.x)
        ampere_patterns = ["A100", "A10", "A30", "A40", "A6000", "RTX 30"]
        
        # Hopper GPUs (compute capability 9.x)
        hopper_patterns = ["H100", "H200", "GH200"]
        
        # Ada Lovelace GPUs (compute capability 8.9)
        ada_patterns = ["L40", "L4", "RTX 40"]
        
        # Blackwell GPUs (compute capability 10.x)
        blackwell_patterns = ["B100", "B200", "GB200"]
        
        all_patterns = ampere_patterns + hopper_patterns + ada_patterns + blackwell_patterns
        
        return any(pattern in gpu_upper for pattern in all_patterns)

    def _infer_gpu_interconnect(self, gpu_type: str, gpu_count: int) -> str:
        """
        Infer GPU interconnect type based on GPU model and count.
        
        This is a best-effort inference:
        - Data center GPUs (A100, H100) typically use NVLink/NVSwitch in multi-GPU configs
        - Consumer GPUs typically use PCIe
        - Single GPU always uses PCIe (no interconnect needed)
        
        Returns:
            Interconnect type: "NVLink", "NVSwitch", or "PCIe"
        """
        if gpu_count <= 1:
            return "PCIe"
        
        gpu_upper = gpu_type.upper()
        
        # High-end data center GPUs with NVSwitch (8+ GPU configs)
        nvswitch_gpus = ["H100", "H200", "GH200", "A100-SXM", "B100", "B200", "GB200"]
        if gpu_count >= 8 and any(pattern in gpu_upper for pattern in nvswitch_gpus):
            return "NVSwitch"
        
        # Data center GPUs with NVLink
        nvlink_gpus = ["A100", "H100", "H200", "V100", "A30", "A40"]
        if any(pattern in gpu_upper for pattern in nvlink_gpus):
            return "NVLink"
        
        # Default to PCIe for consumer GPUs
        return "PCIe"

    def _generate_notes(self, modelspecs: list[ModelSpec]) -> Optional[list[str]]:
        """Generate informational notes."""
        notes = []

        runtime_enabled = any(
            spec.runtime_state and spec.runtime_state.vllm
            for spec in modelspecs
        )

        if not runtime_enabled:
            notes.append("Runtime metrics not collected. Use --collect-runtime for vLLM API metrics.")

        return notes if notes else None
