"""
Scan orchestrator.

Coordinates the scanning process across namespaces, collecting
data from all sources and generating ModelSpec output.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from piqc import __version__
from piqc.collectors.config_collector import ConfigCollector
from piqc.collectors.gpu_collector import GPUCollector, GPUMetrics
from piqc.collectors.vllm_api_client import (
    KubectlExecVLLMClient,
    VLLMAPIClient,
    VLLMRuntimeMetrics,
    discover_vllm_service,
)
from piqc.collectors.vllm_collector import VLLMCollector
from piqc.core.discovery import DeploymentDiscovery
from piqc.core.k8s_client import K8sClient
from piqc.models.modelspec import (
    CollectionMetadata,
    DataCompleteness,
    Endpoint,
    EngineInfo,
    GPUInfo,
    InferenceConfig,
    InferenceDeployment,
    KubernetesMetadata,
    MetadataInfo,
    ModelInfo,
    ModelSpec,
    ResourceInfo,
    RuntimeState,
    VLLMRuntimeState,
)
from piqc.parsers.vllm_parser import VLLMParser
from piqc.utils.exceptions import GPUMetricsUnavailableError
from piqc.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class UnallocatedNodeInfo:
    """A node with GPU capacity that has no pods scheduled on it (or partially unallocated)."""

    node_name: str
    gpu_type: str        # e.g. "A100-SXM4-80GB"
    total_gpus: int      # total GPU capacity on the node
    allocated_gpus: int  # GPUs claimed by running pods
    unallocated_gpus: int


@dataclass
class PendingGPUPod:
    """A pod that is Pending due to insufficient GPU resources.

    Represents active demand that cannot be satisfied given the current
    cluster GPU layout. When paired with fragmented nodes, this signals
    that rebalancing would unblock real workloads.
    """

    pod_name: str
    namespace: str
    gpus_needed: int
    pending_minutes: float   # how long the pod has been waiting


@dataclass
class FragmentedNodeInfo:
    """A node with leftover GPU slots too small to fit any running model.

    Unlike a fully unallocated node, a fragmented node has pods scheduled
    but the remaining free GPUs cannot accommodate any model in the cluster
    (GPUs cannot be combined across nodes). The stranded slots are paid for
    but permanently unusable without rebalancing.
    """

    node_name: str
    gpu_type: str
    total_gpus: int
    allocated_gpus: int
    stranded_gpus: int        # free GPUs that cannot fit any known model
    min_model_gpus_needed: int  # smallest GPU footprint seen in the cluster


class ScanResult:
    """Results from a scan operation."""

    def __init__(self) -> None:
        self.modelspecs: list[ModelSpec] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.duration_seconds: float = 0.0
        self.namespaces_scanned: int = 0
        self.pods_analyzed: int = 0
        self.deployments_found: int = 0
        self.unallocated_nodes: list[UnallocatedNodeInfo] = []
        self.fragmented_nodes: list[FragmentedNodeInfo] = []
        self.pending_gpu_pods: list[PendingGPUPod] = []
    
    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(message)
    
    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        logger.error(message)


class ScanOrchestrator:
    """
    Orchestrates the complete scanning process.
    
    Coordinates discovery, data collection, parsing, and ModelSpec
    generation across all namespaces.
    """
    
    def __init__(
        self,
        k8s_client: K8sClient,
        enable_exec: bool = True,
        enable_logs: bool = True,
        enable_runtime_collection: bool = False,
        workers: int = 5,
        timeout: int = 30,
    ) -> None:
        """
        Initialize the orchestrator.
        
        Args:
            k8s_client: Kubernetes client instance.
            enable_exec: Whether to enable pod exec for GPU metrics.
            enable_logs: Whether to enable log reading.
            enable_runtime_collection: Whether to collect runtime metrics via vLLM API.
            workers: Number of parallel workers for scanning.
            timeout: Timeout for individual operations.
        """
        self.k8s_client = k8s_client
        self.enable_exec = enable_exec
        self.enable_logs = enable_logs
        self.enable_runtime_collection = enable_runtime_collection
        self.workers = workers
        self.timeout = timeout
        
        # Initialize components
        self.discovery = DeploymentDiscovery()
        self.config_collector = ConfigCollector()
        self.vllm_collector = VLLMCollector()
        self.gpu_collector = GPUCollector(k8s_client, exec_timeout=timeout)
        
        self.vllm_parser = VLLMParser()
    
    def scan(
        self,
        namespaces: Optional[list[str]] = None,
    ) -> ScanResult:
        """
        Execute a full cluster scan.
        
        Args:
            namespaces: Specific namespaces to scan. If None, scans all.
            
        Returns:
            ScanResult with all collected data.
        """
        start_time = time.time()
        result = ScanResult()
        
        # Get namespaces to scan
        if namespaces:
            target_namespaces = namespaces
        else:
            try:
                target_namespaces = self.k8s_client.list_namespaces()
            except Exception as e:
                result.add_error(f"Failed to list namespaces: {e}")
                return result
        
        result.namespaces_scanned = len(target_namespaces)
        logger.info(f"Scanning {len(target_namespaces)} namespace(s)")
        
        # Discover all pods
        all_deployments: list[InferenceDeployment] = []
        
        for namespace in target_namespaces:
            try:
                deployments = self._scan_namespace(namespace, result)
                all_deployments.extend(deployments)
            except Exception as e:
                result.add_warning(f"Failed to scan namespace '{namespace}': {e}")
        
        # Group deployments
        grouped = self.discovery.group_deployments(all_deployments)
        result.deployments_found = len(grouped)
        
        logger.info(f"Found {len(grouped)} inference deployment(s)")
        
        # Generate ModelSpecs
        if self.workers > 1 and len(grouped) > 1:
            result.modelspecs = self._process_parallel(grouped, result)
        else:
            result.modelspecs = self._process_sequential(grouped, result)

        # Analyze node-level GPU allocation (unallocated = dark capacity)
        try:
            result.unallocated_nodes = self._analyze_node_capacity()
        except Exception as e:
            result.add_warning(f"Node capacity analysis failed: {e}")

        # Analyze fragmented GPU slots (free slots too small to fit any model)
        try:
            result.fragmented_nodes = self._analyze_fragmentation(
                result.unallocated_nodes, result.modelspecs
            )
        except Exception as e:
            result.add_warning(f"Fragmentation analysis failed: {e}")

        # Detect pending pods waiting for GPU resources
        try:
            result.pending_gpu_pods = self._analyze_pending_gpu_pods()
        except Exception as e:
            result.add_warning(f"Pending GPU pod analysis failed: {e}")

        result.duration_seconds = time.time() - start_time
        
        return result
    
    def _scan_namespace(
        self,
        namespace: str,
        result: ScanResult,
    ) -> list[InferenceDeployment]:
        """Scan a single namespace for inference deployments."""
        logger.debug(f"Scanning namespace: {namespace}")
        
        # Get running pods
        pods = self.k8s_client.list_pods(
            namespace=namespace,
            field_selector="status.phase=Running",
        )
        
        running_pods = self.discovery.filter_running_pods(pods)
        result.pods_analyzed += len(running_pods)
        
        # Analyze each pod
        deployments: list[InferenceDeployment] = []
        
        for pod in running_pods:
            deployment = self.discovery.analyze_pod(pod)
            if deployment:
                deployments.append(deployment)
        
        return deployments
    
    def _process_sequential(
        self,
        deployments: list[InferenceDeployment],
        result: ScanResult,
    ) -> list[ModelSpec]:
        """Process deployments sequentially."""
        modelspecs: list[ModelSpec] = []
        
        for deployment in deployments:
            try:
                modelspec = self._generate_modelspec(deployment, result)
                modelspecs.append(modelspec)
            except Exception as e:
                result.add_warning(
                    f"Failed to process {deployment.namespace}/{deployment.name}: {e}"
                )
        
        return modelspecs
    
    def _process_parallel(
        self,
        deployments: list[InferenceDeployment],
        result: ScanResult,
    ) -> list[ModelSpec]:
        """Process deployments in parallel."""
        modelspecs: list[ModelSpec] = []
        
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            future_to_deployment = {
                executor.submit(self._generate_modelspec, dep, result): dep
                for dep in deployments
            }
            
            for future in as_completed(future_to_deployment):
                deployment = future_to_deployment[future]
                try:
                    modelspec = future.result()
                    modelspecs.append(modelspec)
                except Exception as e:
                    result.add_warning(
                        f"Failed to process {deployment.namespace}/{deployment.name}: {e}"
                    )
        
        return modelspecs
    
    def _generate_modelspec(
        self,
        deployment: InferenceDeployment,
        result: ScanResult,
    ) -> ModelSpec:
        """Generate a ModelSpec for a single deployment."""
        import time as _time
        start_time = _time.time()
        
        logger.debug(f"Generating ModelSpec for {deployment.namespace}/{deployment.name}")
        
        # Initialize tracking flags
        has_gpu_metrics = False
        gpu_infos: list[GPUInfo] = []
        warnings: list[str] = []
        errors: list[str] = []
        
        # If no GPU resource request but FAKE_GPU env vars are present (test workloads),
        # use the env vars to synthesize GPU info directly without exec.
        if deployment.gpu_count == 0 and deployment.env_vars.get("FAKE_GPU_MODEL"):
            fake_count = int(deployment.env_vars.get("FAKE_GPU_COUNT", "1"))
            fake_model = deployment.env_vars["FAKE_GPU_MODEL"]
            fake_util = deployment.env_vars.get("FAKE_GPU_UTILIZATION")
            fake_mem_total = deployment.env_vars.get("FAKE_GPU_MEMORY_TOTAL")
            fake_mem_used = deployment.env_vars.get("FAKE_GPU_MEMORY_USED")
            for i in range(fake_count):
                gpu_infos.append(GPUInfo(
                    index=i,
                    type=fake_model,
                    memory_total=fake_mem_total if fake_mem_total else None,
                    memory_used=fake_mem_used if fake_mem_used else None,
                    utilization=int(fake_util) if fake_util else None,
                ))
            has_gpu_metrics = bool(gpu_infos)

        # Collect GPU metrics if enabled
        if self.enable_exec and deployment.gpu_count > 0:
            for pod_name in deployment.pod_names[:1]:  # Just first pod for now
                try:
                    gpu_metrics = self.gpu_collector.collect(
                        pod_name=pod_name,
                        namespace=deployment.namespace,
                    )
                    if gpu_metrics:
                        has_gpu_metrics = True
                        gpu_infos.extend(
                            self._convert_gpu_metrics(gpu_metrics, pod_name)
                        )
                except GPUMetricsUnavailableError as e:
                    warnings.append(f"{deployment.name}: {e.reason or 'GPU metrics unavailable'}")
                except Exception as e:
                    warnings.append(f"{deployment.name}: GPU collection failed - {e}")

        # Fallback: if no GPU info from nvidia-smi, infer GPU type from node labels
        if not gpu_infos and deployment.gpu_count > 0:
            try:
                pods = self.k8s_client.list_pods(
                    namespace=deployment.namespace,
                    field_selector=f"metadata.name={deployment.pod_names[0]}",
                )
                if pods and pods[0].spec and pods[0].spec.node_name:
                    node_name = pods[0].spec.node_name
                    all_nodes = self.k8s_client.list_nodes()
                    nodes = [n for n in all_nodes if n.metadata.name == node_name]
                    if nodes:
                        node_labels = nodes[0].metadata.labels or {}
                        gpu_type = (
                            node_labels.get("nvidia.com/gpu.product")
                            or node_labels.get("cloud.google.com/gke-accelerator")
                            or node_labels.get("node.kubernetes.io/instance-type")
                            or "unknown"
                        )
                        for i in range(deployment.gpu_count):
                            gpu_infos.append(GPUInfo(
                                index=i,
                                type=gpu_type,
                                memory_total="unknown",
                            ))
                        has_gpu_metrics = bool(gpu_infos)
                        logger.debug(f"Inferred GPU type from node labels: {gpu_type}")
            except Exception as e:
                logger.debug(f"Failed to infer GPU type from node: {e}")
        
        # Parse framework-specific configuration
        model_info, inference_config = self._parse_config(deployment)
        
        # Build endpoints
        container_config = None
        endpoints: list[Endpoint] = []
        
        if deployment.pod_names:
            pods = self.k8s_client.list_pods(
                namespace=deployment.namespace,
                field_selector=f"metadata.name={deployment.pod_names[0]}",
            )
            if pods:
                container_config = self.config_collector.collect_container_config(pods[0])
                if container_config:
                    endpoint_infos = self.config_collector.detect_endpoints(container_config)
                    endpoints = [
                        Endpoint(
                            type=e.type,
                            port=e.port,
                            path=e.path,
                            url=e.url,
                        )
                        for e in endpoint_infos
                    ]
        
        # Build resource info
        resource_info = ResourceInfo(
            replicas=deployment.replicas,
            gpus=gpu_infos,
            gpu_memory_utilization=None,
            cpu_request=deployment.cpu_request,
            cpu_limit=container_config.cpu_limit if container_config else None,
            memory_request=deployment.memory_request,
            memory_limit=container_config.memory_limit if container_config else None,
        )
        
        # Parse image tag
        image = deployment.container_image or ""
        image_parts = image.rsplit(":", 1)
        image_tag = image_parts[1] if len(image_parts) > 1 else None
        
        # Collect runtime metrics via vLLM API
        runtime_state: Optional[RuntimeState] = None
        has_runtime_metrics = False
        
        if self.enable_runtime_collection and deployment.framework == "vllm":
            for pod_name in deployment.pod_names[:1]:
                try:
                    service_info = discover_vllm_service(
                        self.k8s_client,
                        deployment.namespace,
                        pod_name,
                    )
                    if service_info:
                        access_method, port = service_info
                        
                        # Choose appropriate API client based on access method
                        if access_method == "kubectl-exec":
                            api_client = KubectlExecVLLMClient(
                                k8s_client=self.k8s_client,
                                namespace=deployment.namespace,
                                pod_name=pod_name,
                                port=port,
                                timeout=self.timeout,
                            )
                            collection_method = "vllm-api-kubectl-exec"
                        else:
                            # Direct HTTP access
                            api_client = VLLMAPIClient(access_method, timeout=self.timeout)
                            collection_method = "vllm-api"
                        
                        vllm_metrics = api_client.get_runtime_metrics()
                        logger.debug(f"Runtime metrics api_available={vllm_metrics.api_available} health={vllm_metrics.health_status} model={vllm_metrics.loaded_model}")

                        if vllm_metrics.api_available:
                            has_runtime_metrics = True
                            runtime_state = RuntimeState(
                                collection_method=collection_method,
                                vllm=VLLMRuntimeState(
                                    loaded_model=vllm_metrics.loaded_model,
                                    requests_running=vllm_metrics.requests.running,
                                    requests_waiting=vllm_metrics.requests.waiting,
                                    requests_swapped=vllm_metrics.requests.swapped,
                                    ttft_p50=vllm_metrics.latency.ttft_p50,
                                    ttft_p95=vllm_metrics.latency.ttft_p95,
                                    ttft_p99=vllm_metrics.latency.ttft_p99,
                                    tpot_p50=vllm_metrics.latency.tpot_p50,
                                    tpot_p95=vllm_metrics.latency.tpot_p95,
                                    e2e_p50=vllm_metrics.latency.e2e_p50,
                                    e2e_p95=vllm_metrics.latency.e2e_p95,
                                    e2e_p99=vllm_metrics.latency.e2e_p99,
                                    prompt_tokens_per_sec=vllm_metrics.throughput.prompt_tokens_per_second,
                                    generation_tokens_per_sec=vllm_metrics.throughput.generation_tokens_per_second,
                                    prompt_tokens_total=vllm_metrics.throughput.prompt_tokens_total,
                                    generation_tokens_total=vllm_metrics.throughput.generation_tokens_total,
                                    gpu_cache_usage_percent=vllm_metrics.cache.gpu_cache_usage_percent,
                                    cpu_cache_usage_percent=vllm_metrics.cache.cpu_cache_usage_percent,
                                    prefix_cache_hit_rate=vllm_metrics.cache.prefix_cache_hit_rate,
                                    api_available=True,
                                    health_status=vllm_metrics.health_status,
                                    collection_timestamp=vllm_metrics.collection_timestamp.isoformat() + "Z",
                                ),
                            )
                        else:
                            warnings.append(f"{deployment.name}: vLLM API not available")
                    else:
                        warnings.append(f"{deployment.name}: Could not discover vLLM service endpoint")
                except Exception as e:
                    warnings.append(f"{deployment.name}: Runtime metrics collection failed - {e}")
        
        # Calculate duration
        duration_seconds = _time.time() - start_time
        if duration_seconds < 1:
            duration_str = f"{duration_seconds * 1000:.0f}ms"
        else:
            duration_str = f"{duration_seconds:.2f}s"
        
        # Build ModelSpec
        modelspec = ModelSpec(
            metadata=MetadataInfo(
                name=deployment.name,
                namespace=deployment.namespace,
                labels=deployment.labels,
                annotations={},
                collection_timestamp=datetime.utcnow().isoformat() + "Z",
                collector_version=__version__,
            ),
            model=model_info,
            engine=EngineInfo(
                name=deployment.framework,
                version=None,
                framework=None,
                detection_confidence=deployment.confidence,
            ),
            inference=inference_config,
            resources=resource_info,
            endpoints=endpoints,
            kubernetes=KubernetesMetadata(
                deployment_type=deployment.deployment_type,
                cluster_name=self.k8s_client.cluster_name,
                creation_timestamp=None,
                image=image,
                image_tag=image_tag,
            ),
            runtime_state=runtime_state,
            collection=CollectionMetadata(
                mode="incluster" if self.k8s_client.context == "in-cluster" else "remote",
                duration=duration_str,
                warnings=warnings,
                errors=errors,
                data_completeness=DataCompleteness(
                    static_config=True,
                    gpu_metrics=has_gpu_metrics,
                    runtime_metrics=has_runtime_metrics,
                ),
            ),
        )
        
        return modelspec
    
    def _parse_config(
        self,
        deployment: InferenceDeployment,
    ) -> tuple[ModelInfo, InferenceConfig]:
        """Parse configuration based on detected framework."""
        if deployment.framework == "vllm":
            vllm_config = self.vllm_collector.collect(
                env_vars=deployment.env_vars,
                container_args=deployment.container_args,
            )
            model_info = self.vllm_parser.parse_model_info(vllm_config)
            inference_config = self.vllm_parser.parse_inference_config(vllm_config)
            return model_info, inference_config
        
        # Unknown framework - return minimal info
        return (
            ModelInfo(
                name=None,
                source=None,
                confidence=0.0,
                identification_method="none",
            ),
            InferenceConfig(),
        )
    
    def _convert_gpu_metrics(
        self,
        metrics: list[GPUMetrics],
        pod_name: str,
    ) -> list[GPUInfo]:
        """Convert GPU metrics to GPUInfo models."""
        return [
            GPUInfo(
                index=m.gpu_index,
                type=m.gpu_model,
                memory_total=m.to_memory_str(m.memory_total_mb),
                memory_used=m.to_memory_str(m.memory_used_mb) if m.memory_used_mb else None,
                utilization=m.utilization_percent if m.utilization_percent else None,
                temperature=m.temperature_celsius if m.temperature_celsius else None,
                power_draw=m.power_draw_watts if m.power_draw_watts else None,
                pod_name=pod_name,
            )
            for m in metrics
        ]

    def _analyze_node_capacity(self) -> list[UnallocatedNodeInfo]:
        """
        Compare node GPU capacity against allocated GPU requests.

        Returns nodes where GPU capacity exceeds what running pods have claimed,
        representing dark/unallocated capacity that is still being paid for.
        """
        nodes = self.k8s_client.list_nodes()
        if not nodes:
            return []

        # Sum GPU requests per node across all running pods (all namespaces)
        allocated_per_node: dict[str, int] = {}
        try:
            all_pods = self.k8s_client.list_all_pods(
                field_selector="status.phase=Running",
            )
        except Exception:
            all_pods = []

        for pod in all_pods:
            node_name = pod.spec.node_name if pod.spec else None
            if not node_name:
                continue
            for container in (pod.spec.containers or []):
                if not container.resources or not container.resources.requests:
                    continue
                gpu_req = container.resources.requests.get("nvidia.com/gpu", "0")
                try:
                    allocated_per_node[node_name] = (
                        allocated_per_node.get(node_name, 0) + int(gpu_req)
                    )
                except ValueError:
                    pass

        unallocated: list[UnallocatedNodeInfo] = []

        for node in nodes:
            if not node.status or not node.status.allocatable:
                continue

            total_gpus = int(node.status.allocatable.get("nvidia.com/gpu", "0"))
            if total_gpus == 0:
                continue

            node_name = node.metadata.name
            allocated = allocated_per_node.get(node_name, 0)
            free = total_gpus - allocated

            if free > 0:
                # Detect GPU type from node labels (best-effort)
                labels = node.metadata.labels or {}
                gpu_type = (
                    labels.get("nvidia.com/gpu.product")
                    or labels.get("cloud.google.com/gke-accelerator")
                    or labels.get("node.kubernetes.io/instance-type")
                    or "unknown"
                )
                unallocated.append(
                    UnallocatedNodeInfo(
                        node_name=node_name,
                        gpu_type=gpu_type,
                        total_gpus=total_gpus,
                        allocated_gpus=allocated,
                        unallocated_gpus=free,
                    )
                )

        return unallocated

    def _analyze_fragmentation(
        self,
        unallocated_nodes: list[UnallocatedNodeInfo],
        modelspecs: list[ModelSpec],
    ) -> list[FragmentedNodeInfo]:
        """Identify nodes with stranded GPU slots that cannot fit any running model.

        A fragmented node has free GPU slots, but the count is smaller than
        the minimum GPU footprint of any GPU workload in the cluster. Because
        GPUs cannot be shared across nodes, those slots are permanently unusable
        until the node is rebalanced.

        Args:
            unallocated_nodes: Nodes with at least one free GPU slot (from
                _analyze_node_capacity).
            modelspecs: Discovered vLLM model deployments (used as one source
                of GPU footprints; all running GPU pods are also consulted).

        Returns:
            List of FragmentedNodeInfo for nodes with stranded slots.
        """
        if not unallocated_nodes:
            return []

        # Derive minimum GPU count from ALL running pods requesting GPUs —
        # not just discovered vLLM modelspecs. This catches any GPU workload
        # in the cluster, including non-vLLM frameworks.
        gpu_footprints: list[int] = []

        try:
            all_pods = self.k8s_client.list_all_pods(
                field_selector="status.phase=Running",
            )
            for pod in all_pods:
                pod_gpus = 0
                for container in (pod.spec.containers if pod.spec else []) or []:
                    if not container.resources or not container.resources.requests:
                        continue
                    gpu_req = container.resources.requests.get("nvidia.com/gpu", "0")
                    try:
                        pod_gpus += int(gpu_req)
                    except ValueError:
                        pass
                if pod_gpus > 0:
                    gpu_footprints.append(pod_gpus)
        except Exception as e:
            logger.debug(f"Could not list pods for fragmentation analysis: {e}")

        # Also include modelspec GPU counts as a fallback / supplement.
        for spec in modelspecs:
            if spec.resources and spec.resources.gpus:
                count = len(spec.resources.gpus)
                if count > 0:
                    gpu_footprints.append(count)

        if not gpu_footprints:
            # No GPU workloads found — cannot determine fragmentation threshold.
            logger.debug("Fragmentation analysis skipped: no GPU workloads found in cluster")
            return []

        min_model_gpus = min(gpu_footprints)
        logger.debug(f"Fragmentation threshold: min model GPU footprint = {min_model_gpus}")

        fragmented: list[FragmentedNodeInfo] = []
        for node in unallocated_nodes:
            if node.unallocated_gpus < min_model_gpus:
                # Free slots exist but cannot accommodate even the smallest model.
                fragmented.append(
                    FragmentedNodeInfo(
                        node_name=node.node_name,
                        gpu_type=node.gpu_type,
                        total_gpus=node.total_gpus,
                        allocated_gpus=node.allocated_gpus,
                        stranded_gpus=node.unallocated_gpus,
                        min_model_gpus_needed=min_model_gpus,
                    )
                )
                logger.debug(
                    f"Fragmented node {node.node_name}: "
                    f"{node.unallocated_gpus} free GPU(s) < {min_model_gpus} needed"
                )

        return fragmented

    def _analyze_pending_gpu_pods(self) -> list[PendingGPUPod]:
        """Find pods that are Pending due to insufficient GPU resources.

        Reads all Pending pods and checks whether they are requesting GPUs.
        These represent active demand that the cluster cannot currently satisfy —
        the strongest signal that fragmentation or capacity is blocking real work.

        Returns:
            List of PendingGPUPod for pods waiting for GPU resources.
        """
        from datetime import timezone

        try:
            all_pods = self.k8s_client.list_all_pods(
                field_selector="status.phase=Pending",
            )
        except Exception as e:
            logger.debug(f"Could not list pending pods: {e}")
            return []

        pending: list[PendingGPUPod] = []
        now = datetime.now(timezone.utc)

        for pod in all_pods:
            if not pod.spec:
                continue

            # Sum GPU requests across all containers in this pod
            gpus_needed = 0
            for container in (pod.spec.containers or []):
                if not container.resources or not container.resources.requests:
                    continue
                gpu_req = container.resources.requests.get("nvidia.com/gpu", "0")
                try:
                    gpus_needed += int(gpu_req)
                except ValueError:
                    pass

            if gpus_needed == 0:
                continue

            # Calculate how long the pod has been pending
            pending_minutes = 0.0
            if pod.metadata and pod.metadata.creation_timestamp:
                created_at = pod.metadata.creation_timestamp
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
                pending_minutes = (now - created_at).total_seconds() / 60.0

            # Use owner reference (deployment/replicaset name) if available,
            # otherwise strip the two hash suffixes from the pod name.
            raw_name = pod.metadata.name if pod.metadata else "unknown"
            owner_name = None
            if pod.metadata and pod.metadata.owner_references:
                for ref in pod.metadata.owner_references:
                    if ref.kind == "ReplicaSet" and ref.name:
                        # ReplicaSet name is deployment-name + one hash — strip it
                        parts = ref.name.rsplit("-", 1)
                        owner_name = parts[0] if len(parts) > 1 else ref.name
                        break
            pod_name = owner_name or raw_name
            namespace = pod.metadata.namespace if pod.metadata else "unknown"

            pending.append(
                PendingGPUPod(
                    pod_name=pod_name,
                    namespace=namespace,
                    gpus_needed=gpus_needed,
                    pending_minutes=pending_minutes,
                )
            )
            logger.debug(
                f"Pending GPU pod {namespace}/{pod_name}: "
                f"needs {gpus_needed} GPU(s), waiting {pending_minutes:.1f} min"
            )

        return pending
