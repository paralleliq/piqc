"""
Custom exception hierarchy for piqc.

This module defines all custom exceptions used throughout the application,
providing clear error categorization and meaningful error messages.
"""

from typing import Optional


class ModelSpecError(Exception):
    """
    Base exception for all piqc errors.
    
    All custom exceptions in this application inherit from this class,
    allowing for broad exception catching when needed.
    """

    def __init__(self, message: str, details: Optional[str] = None) -> None:
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message.
            details: Optional additional details for debugging.
        """
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} - {self.details}"
        return self.message


class KubernetesConnectionError(ModelSpecError):
    """
    Raised when connection to the Kubernetes cluster fails.
    
    This can occur due to:
    - Invalid or missing kubeconfig
    - Network connectivity issues
    - Cluster unavailability
    - Invalid context specified
    """

    def __init__(
        self,
        message: str = "Failed to connect to Kubernetes cluster",
        details: Optional[str] = None,
        context: Optional[str] = None,
    ) -> None:
        self.context = context
        if context and not details:
            details = f"Context: {context}"
        super().__init__(message, details)


class RBACPermissionError(ModelSpecError):
    """
    Raised when the service account lacks required RBAC permissions.
    
    This typically indicates that the user or service account running
    the introspector does not have sufficient permissions to perform
    the requested operations.
    """

    def __init__(
        self,
        message: str = "Insufficient RBAC permissions",
        resource: Optional[str] = None,
        verb: Optional[str] = None,
        namespace: Optional[str] = None,
    ) -> None:
        self.resource = resource
        self.verb = verb
        self.namespace = namespace
        
        details_parts = []
        if resource:
            details_parts.append(f"Resource: {resource}")
        if verb:
            details_parts.append(f"Verb: {verb}")
        if namespace:
            details_parts.append(f"Namespace: {namespace}")
        
        details = ", ".join(details_parts) if details_parts else None
        super().__init__(message, details)


class GPUMetricsUnavailableError(ModelSpecError):
    """
    Raised when GPU metrics collection fails.
    
    This is a non-fatal error - the scan will continue with degraded
    GPU information. Common causes:
    - nvidia-smi not installed in container
    - Pod exec permissions denied
    - No NVIDIA GPUs present
    """

    def __init__(
        self,
        message: str = "GPU metrics unavailable",
        pod_name: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> None:
        self.pod_name = pod_name
        self.reason = reason
        
        details_parts = []
        if pod_name:
            details_parts.append(f"Pod: {pod_name}")
        if reason:
            details_parts.append(f"Reason: {reason}")
        
        details = ", ".join(details_parts) if details_parts else None
        super().__init__(message, details)


class ConfigurationParseError(ModelSpecError):
    """
    Raised when parsing configuration data fails.
    
    This can occur when:
    - Environment variables contain invalid values
    - CLI arguments cannot be parsed
    - Configuration format is unexpected
    """

    def __init__(
        self,
        message: str = "Failed to parse configuration",
        source: Optional[str] = None,
        field: Optional[str] = None,
    ) -> None:
        self.source = source
        self.field = field
        
        details_parts = []
        if source:
            details_parts.append(f"Source: {source}")
        if field:
            details_parts.append(f"Field: {field}")
        
        details = ", ".join(details_parts) if details_parts else None
        super().__init__(message, details)


class PodExecutionError(ModelSpecError):
    """
    Raised when executing a command in a pod fails.
    
    This is typically non-fatal and results in degraded data collection.
    """

    def __init__(
        self,
        message: str = "Pod command execution failed",
        pod_name: Optional[str] = None,
        namespace: Optional[str] = None,
        command: Optional[str] = None,
        exit_code: Optional[int] = None,
    ) -> None:
        self.pod_name = pod_name
        self.namespace = namespace
        self.command = command
        self.exit_code = exit_code
        
        details_parts = []
        if pod_name:
            details_parts.append(f"Pod: {pod_name}")
        if namespace:
            details_parts.append(f"Namespace: {namespace}")
        if exit_code is not None:
            details_parts.append(f"Exit code: {exit_code}")
        
        details = ", ".join(details_parts) if details_parts else None
        super().__init__(message, details)


class InvalidModelSpecError(ModelSpecError):
    """
    Raised when a generated ModelSpec fails validation.
    
    This indicates a bug in the generation logic or unexpected
    data from the cluster.
    """

    def __init__(
        self,
        message: str = "Generated ModelSpec is invalid",
        validation_errors: Optional[list[str]] = None,
    ) -> None:
        self.validation_errors = validation_errors or []
        
        details = "; ".join(self.validation_errors) if self.validation_errors else None
        super().__init__(message, details)


class TimeoutError(ModelSpecError):
    """
    Raised when an operation times out.
    """

    def __init__(
        self,
        message: str = "Operation timed out",
        operation: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ) -> None:
        self.operation = operation
        self.timeout_seconds = timeout_seconds
        
        details_parts = []
        if operation:
            details_parts.append(f"Operation: {operation}")
        if timeout_seconds is not None:
            details_parts.append(f"Timeout: {timeout_seconds}s")
        
        details = ", ".join(details_parts) if details_parts else None
        super().__init__(message, details)
