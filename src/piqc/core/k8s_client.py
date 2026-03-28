"""
Kubernetes API client wrapper.

Provides a secure, reliable interface to Kubernetes clusters with
proper error handling, retry logic, and connection management.
"""

import time
from pathlib import Path
from typing import Any, Callable, Optional

from kubernetes import client, config
from kubernetes.client import (
    CoreV1Api,
    AppsV1Api,
    V1Deployment,
    V1StatefulSet,
    V1Node,
    V1Pod,
    V1PodList,
)
from kubernetes.client.exceptions import ApiException
from kubernetes.stream import stream

from piqc.utils.exceptions import (
    KubernetesConnectionError,
    RBACPermissionError,
    PodExecutionError,
)
from piqc.utils.logger import get_logger


logger = get_logger(__name__)


class K8sClient:
    """
    Kubernetes API client wrapper with retry logic and error handling.
    
    This class provides a simplified interface to the Kubernetes API
    with built-in support for:
    - Multiple configuration sources (kubeconfig, in-cluster)
    - Automatic retry with exponential backoff
    - Graceful error handling with meaningful messages
    - Read-only operations only (production-safe)
    
    Attributes:
        core_api: Kubernetes Core V1 API client.
        apps_api: Kubernetes Apps V1 API client.
        context: The Kubernetes context being used.
        cluster_name: Name of the connected cluster (if available).
    """
    
    # Retry configuration
    MAX_RETRIES = 3
    INITIAL_BACKOFF_SECONDS = 1.0
    MAX_BACKOFF_SECONDS = 10.0
    BACKOFF_MULTIPLIER = 2.0
    
    # Default timeouts
    DEFAULT_TIMEOUT_SECONDS = 30
    
    def __init__(
        self,
        kubeconfig_path: Optional[str] = None,
        context: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """
        Initialize the Kubernetes client.
        
        Args:
            kubeconfig_path: Path to kubeconfig file. If None, uses default
                locations (~/.kube/config) or in-cluster config.
            context: Kubernetes context to use. If None, uses current context.
            timeout: Default timeout for API operations in seconds.
            
        Raises:
            KubernetesConnectionError: If connection cannot be established.
        """
        self.timeout = timeout
        self.context = context
        self.cluster_name: Optional[str] = None
        self._kubeconfig_path = kubeconfig_path
        
        self._load_config()
        
        # Initialize API clients
        self.core_api = CoreV1Api()
        self.apps_api = AppsV1Api()
        
        logger.debug(f"Kubernetes client initialized for context: {self.context}")
    
    def _load_config(self) -> None:
        """
        Load Kubernetes configuration from available sources.
        
        Attempts to load configuration in the following order:
        1. Specified kubeconfig path
        2. Default kubeconfig location
        3. In-cluster configuration
        
        Raises:
            KubernetesConnectionError: If no valid configuration found.
        """
        # Try kubeconfig first
        kubeconfig_path = self._kubeconfig_path
        if kubeconfig_path is None:
            default_path = Path.home() / ".kube" / "config"
            if default_path.exists():
                kubeconfig_path = str(default_path)
        
        if kubeconfig_path:
            try:
                config.load_kube_config(
                    config_file=kubeconfig_path,
                    context=self.context,
                )
                
                # Extract context information
                contexts, active_context = config.list_kube_config_contexts(
                    config_file=kubeconfig_path
                )
                
                if self.context:
                    # Find specified context
                    matching = [c for c in contexts if c["name"] == self.context]
                    if not matching:
                        available = [c["name"] for c in contexts]
                        raise KubernetesConnectionError(
                            f"Context '{self.context}' not found",
                            details=f"Available contexts: {', '.join(available)}",
                            context=self.context,
                        )
                    self.cluster_name = matching[0].get("context", {}).get("cluster")
                else:
                    self.context = active_context["name"]
                    self.cluster_name = active_context.get("context", {}).get("cluster")
                
                logger.debug(f"Loaded kubeconfig from: {kubeconfig_path}")
                return
                
            except config.ConfigException as e:
                logger.debug(f"Failed to load kubeconfig: {e}")
        
        # Try in-cluster config
        try:
            config.load_incluster_config()
            self.context = "in-cluster"
            self.cluster_name = "in-cluster"
            logger.debug("Loaded in-cluster configuration")
            return
        except config.ConfigException as e:
            logger.debug(f"Failed to load in-cluster config: {e}")
        
        # No valid configuration found
        raise KubernetesConnectionError(
            "Cannot connect to Kubernetes cluster",
            details="No valid kubeconfig found and not running in-cluster",
        )
    
    def test_connection(self) -> bool:
        """
        Test connectivity to the Kubernetes cluster.
        
        Returns:
            True if connection is successful.
            
        Raises:
            KubernetesConnectionError: If connection test fails.
        """
        try:
            self.core_api.get_api_resources(_request_timeout=self.timeout)
            logger.debug("Cluster connection test successful")
            return True
        except ApiException as e:
            if e.status == 401:
                raise KubernetesConnectionError(
                    "Authentication failed",
                    details="Check credentials in kubeconfig",
                    context=self.context,
                )
            elif e.status == 403:
                raise RBACPermissionError(
                    "Access denied to cluster API",
                    resource="api-resources",
                    verb="get",
                )
            else:
                raise KubernetesConnectionError(
                    f"Connection test failed: {e.reason}",
                    context=self.context,
                )
        except Exception as e:
            raise KubernetesConnectionError(
                "Connection test failed",
                details=str(e),
                context=self.context,
            )
    
    def _retry_operation(
        self,
        operation: Callable[[], Any],
        operation_name: str,
    ) -> Any:
        """
        Execute an operation with retry logic.
        
        Args:
            operation: Callable to execute.
            operation_name: Name for logging purposes.
            
        Returns:
            Result of the operation.
            
        Raises:
            The last exception if all retries fail.
        """
        last_exception: Optional[Exception] = None
        backoff = self.INITIAL_BACKOFF_SECONDS
        
        for attempt in range(self.MAX_RETRIES):
            try:
                return operation()
            except ApiException as e:
                last_exception = e
                
                # Don't retry client errors (4xx) except rate limiting
                if 400 <= e.status < 500 and e.status != 429:
                    raise
                
                if attempt < self.MAX_RETRIES - 1:
                    logger.debug(
                        f"{operation_name} failed (attempt {attempt + 1}/{self.MAX_RETRIES}), "
                        f"retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF_SECONDS)
            except Exception as e:
                last_exception = e
                if attempt < self.MAX_RETRIES - 1:
                    logger.debug(
                        f"{operation_name} failed (attempt {attempt + 1}/{self.MAX_RETRIES}), "
                        f"retrying in {backoff:.1f}s"
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * self.BACKOFF_MULTIPLIER, self.MAX_BACKOFF_SECONDS)
        
        if last_exception:
            raise last_exception
        raise RuntimeError(f"{operation_name} failed after {self.MAX_RETRIES} attempts")
    
    def list_namespaces(self) -> list[str]:
        """
        List all accessible namespaces in the cluster.
        
        Returns:
            List of namespace names.
            
        Raises:
            RBACPermissionError: If lacking namespace list permissions.
            KubernetesConnectionError: If the operation fails.
        """
        def _list() -> Any:
            return self.core_api.list_namespace(_request_timeout=self.timeout)
        
        try:
            result = self._retry_operation(_list, "list_namespaces")
            namespaces = [ns.metadata.name for ns in result.items]
            logger.debug(f"Found {len(namespaces)} namespaces")
            return namespaces
        except ApiException as e:
            if e.status == 403:
                raise RBACPermissionError(
                    "Cannot list namespaces",
                    resource="namespaces",
                    verb="list",
                )
            raise KubernetesConnectionError(
                f"Failed to list namespaces: {e.reason}",
                context=self.context,
            )
    
    def list_deployments(
        self,
        namespace: str,
        label_selector: Optional[str] = None,
    ) -> list[V1Deployment]:
        """
        List deployments in a namespace.
        
        Args:
            namespace: Target namespace.
            label_selector: Optional label selector to filter deployments.
            
        Returns:
            List of V1Deployment objects.
            
        Raises:
            RBACPermissionError: If lacking deployment list permissions.
        """
        def _list() -> list[V1Deployment]:
            return self.apps_api.list_namespaced_deployment(
                namespace=namespace,
                label_selector=label_selector,
                _request_timeout=self.timeout,
            )
        
        try:
            result = self._retry_operation(_list, f"list_deployments({namespace})")
            logger.debug(f"Found {len(result.items)} deployments in {namespace}")
            return result.items
        except ApiException as e:
            if e.status == 403:
                raise RBACPermissionError(
                    f"Cannot list deployments in namespace '{namespace}'",
                    resource="deployments",
                    verb="list",
                    namespace=namespace,
                )
            logger.warning(f"Failed to list deployments in {namespace}: {e.reason}")
            return []
    
    def list_statefulsets(
        self,
        namespace: str,
        label_selector: Optional[str] = None,
    ) -> list[V1StatefulSet]:
        """
        List statefulsets in a namespace.
        
        Args:
            namespace: Target namespace.
            label_selector: Optional label selector to filter statefulsets.
            
        Returns:
            List of V1StatefulSet objects.
        """
        def _list() -> list[V1StatefulSet]:
            return self.apps_api.list_namespaced_stateful_set(
                namespace=namespace,
                label_selector=label_selector,
                _request_timeout=self.timeout,
            )
        
        try:
            result = self._retry_operation(_list, f"list_statefulsets({namespace})")
            logger.debug(f"Found {len(result.items)} statefulsets in {namespace}")
            return result.items
        except ApiException as e:
            if e.status == 403:
                raise RBACPermissionError(
                    f"Cannot list statefulsets in namespace '{namespace}'",
                    resource="statefulsets",
                    verb="list",
                    namespace=namespace,
                )
            logger.warning(f"Failed to list statefulsets in {namespace}: {e.reason}")
            return []
    
    def list_pods(
        self,
        namespace: str,
        label_selector: Optional[str] = None,
        field_selector: Optional[str] = None,
    ) -> list[V1Pod]:
        """
        List pods in a namespace.
        
        Args:
            namespace: Target namespace.
            label_selector: Optional label selector to filter pods.
            field_selector: Optional field selector to filter pods.
            
        Returns:
            List of V1Pod objects.
        """
        def _list() -> V1PodList:
            kwargs = {
                "namespace": namespace,
                "_request_timeout": self.timeout,
            }
            if label_selector:
                kwargs["label_selector"] = label_selector
            if field_selector:
                kwargs["field_selector"] = field_selector
            return self.core_api.list_namespaced_pod(**kwargs)
        
        try:
            result = self._retry_operation(_list, f"list_pods({namespace})")
            logger.debug(f"Found {len(result.items)} pods in {namespace}")
            return result.items
        except ApiException as e:
            if e.status == 403:
                raise RBACPermissionError(
                    f"Cannot list pods in namespace '{namespace}'",
                    resource="pods",
                    verb="list",
                    namespace=namespace,
                )
            logger.warning(f"Failed to list pods in {namespace}: {e.reason}")
            return []
    
    def exec_in_pod(
        self,
        pod_name: str,
        namespace: str,
        command: list[str],
        container: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> tuple[str, str]:
        """
        Execute a command inside a pod.
        
        Args:
            pod_name: Name of the target pod.
            namespace: Namespace of the pod.
            command: Command to execute as a list of strings.
            container: Specific container name. If None, uses first container.
            timeout: Command timeout in seconds.
            
        Returns:
            Tuple of (stdout, stderr) strings.
            
        Raises:
            PodExecutionError: If command execution fails.
            RBACPermissionError: If lacking exec permissions.
        """
        exec_timeout = timeout or self.timeout
        
        try:
            resp = stream(
                self.core_api.connect_get_namespaced_pod_exec,
                pod_name,
                namespace,
                command=command,
                container=container,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _preload_content=False,
            )
            
            stdout_data = []
            stderr_data = []
            
            # Read with timeout
            start_time = time.time()
            while resp.is_open():
                if time.time() - start_time > exec_timeout:
                    resp.close()
                    raise PodExecutionError(
                        "Command execution timed out",
                        pod_name=pod_name,
                        namespace=namespace,
                        command=" ".join(command),
                    )
                
                resp.update(timeout=1)
                if resp.peek_stdout():
                    stdout_data.append(resp.read_stdout())
                if resp.peek_stderr():
                    stderr_data.append(resp.read_stderr())
            
            stdout = "".join(stdout_data)
            stderr = "".join(stderr_data)
            
            logger.debug(f"Executed command in {namespace}/{pod_name}")
            return stdout, stderr
            
        except ApiException as e:
            if e.status == 403:
                raise RBACPermissionError(
                    f"Cannot exec in pod '{pod_name}'",
                    resource="pods/exec",
                    verb="create",
                    namespace=namespace,
                )
            raise PodExecutionError(
                f"Failed to exec in pod: {e.reason}",
                pod_name=pod_name,
                namespace=namespace,
                command=" ".join(command),
            )
        except Exception as e:
            raise PodExecutionError(
                f"Failed to exec in pod: {str(e)}",
                pod_name=pod_name,
                namespace=namespace,
                command=" ".join(command),
            )
    
    def get_pod_logs(
        self,
        pod_name: str,
        namespace: str,
        container: Optional[str] = None,
        tail_lines: int = 100,
    ) -> str:
        """
        Get logs from a pod.
        
        Args:
            pod_name: Name of the target pod.
            namespace: Namespace of the pod.
            container: Specific container name. If None, uses first container.
            tail_lines: Number of lines to retrieve from the end.
            
        Returns:
            Log content as a string.
        """
        try:
            logs = self.core_api.read_namespaced_pod_log(
                name=pod_name,
                namespace=namespace,
                container=container,
                tail_lines=tail_lines,
                _request_timeout=self.timeout,
            )
            logger.debug(f"Retrieved logs from {namespace}/{pod_name}")
            return logs
        except ApiException as e:
            if e.status == 403:
                logger.warning(f"Cannot read logs from {namespace}/{pod_name}: permission denied")
            else:
                logger.warning(f"Cannot read logs from {namespace}/{pod_name}: {e.reason}")
            return ""
        except Exception as e:
            logger.warning(f"Cannot read logs from {namespace}/{pod_name}: {str(e)}")
            return ""
    
    def list_all_pods(
        self,
        field_selector: Optional[str] = None,
    ) -> list[V1Pod]:
        """
        List pods across all namespaces.

        Args:
            field_selector: Optional field selector to filter pods.

        Returns:
            List of V1Pod objects.
        """
        def _list() -> V1PodList:
            kwargs: dict[str, Any] = {"_request_timeout": self.timeout}
            if field_selector:
                kwargs["field_selector"] = field_selector
            return self.core_api.list_pod_for_all_namespaces(**kwargs)

        try:
            result = self._retry_operation(_list, "list_all_pods")
            logger.debug(f"Found {len(result.items)} pods across all namespaces")
            return result.items
        except ApiException as e:
            if e.status == 403:
                raise RBACPermissionError(
                    "Cannot list pods across namespaces",
                    resource="pods",
                    verb="list",
                )
            logger.warning(f"Failed to list pods: {e.reason}")
            return []

    def list_nodes(self) -> list[V1Node]:
        """
        List all nodes in the cluster.

        Returns:
            List of V1Node objects.

        Raises:
            RBACPermissionError: If lacking node list permissions.
        """
        def _list() -> Any:
            return self.core_api.list_node(_request_timeout=self.timeout)

        try:
            result = self._retry_operation(_list, "list_nodes")
            logger.debug(f"Found {len(result.items)} nodes")
            return result.items
        except ApiException as e:
            if e.status == 403:
                raise RBACPermissionError(
                    "Cannot list nodes",
                    resource="nodes",
                    verb="list",
                )
            logger.warning(f"Failed to list nodes: {e.reason}")
            return []

    def get_connection_info(self) -> dict[str, str]:
        """
        Get information about the current connection.
        
        Returns:
            Dictionary with connection details.
        """
        return {
            "context": self.context or "unknown",
            "cluster": self.cluster_name or "unknown",
            "timeout": str(self.timeout),
        }
