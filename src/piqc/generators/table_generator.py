"""
Table output generator.

Generates formatted console tables from ModelSpec data.
"""

from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from piqc.models.modelspec import ModelSpec
from piqc.utils.logger import get_logger


logger = get_logger(__name__)


# On-demand cloud GPU cost estimates (USD/hr per GPU).
# Users can override with --gpu-cost.
_GPU_COST_PER_HOUR: dict[str, float] = {
    "H100-SXM5-80GB": 5.00,
    "H100-NVL": 4.50,
    "H100-SXM4-80GB": 4.25,
    "H100-PCIE-80GB": 4.00,
    "H100": 4.00,
    "A100-SXM4-80GB": 3.50,
    "A100-PCIE-80GB": 3.00,
    "A100-SXM4-40GB": 2.50,
    "A100-PCIE-40GB": 2.25,
    "A100": 3.00,
    "L40S": 2.00,
    "A10G": 1.25,
    "A10": 1.00,
    "L4": 0.80,
    "V100-SXM2-32GB": 1.25,
    "V100-SXM2-16GB": 1.00,
    "V100": 1.00,
    "T4": 0.45,
}
_DEFAULT_GPU_COST_PER_HOUR = 2.00
_IDLE_THRESHOLD = 60  # utilization % below which a GPU is considered idle


class TableGenerator:
    """
    Generates formatted console tables from ModelSpec data.
    
    Uses Rich library for professional terminal output.
    """
    
    def __init__(self, console: Optional[Console] = None) -> None:
        """
        Initialize the table generator.
        
        Args:
            console: Rich console instance. Creates new one if not provided.
        """
        self.console = console or Console()
    
    def generate_summary_table(
        self,
        modelspecs: list[ModelSpec],
        gpu_cost_override: Optional[float] = None,
    ) -> None:
        """
        Generate and print a summary table of all ModelSpecs.

        Args:
            modelspecs: List of ModelSpec objects.
            gpu_cost_override: Optional $/GPU/hr override (uses built-in lookup if None).
        """
        if not modelspecs:
            self.console.print("[dim]No inference deployments found[/dim]")
            return

        table = Table(
            title="Discovered Inference Deployments",
            show_header=True,
            header_style="bold",
        )

        # Add columns
        table.add_column("Deployment", style="cyan", no_wrap=True)
        table.add_column("Engine", style="green")
        table.add_column("GPU", style="yellow")
        table.add_column("Replicas", justify="right")
        table.add_column("GPU Util", justify="right")
        table.add_column("$/hr", justify="right")
        table.add_column("Idle $/day", justify="right")
        table.add_column("Namespace", style="dim")

        for spec in modelspecs:
            model_name = spec.model.name or spec.metadata.name
            if len(model_name) > 30:
                model_name = model_name[:27] + "..."

            gpu_info = self._format_gpu_info(spec)
            gpu_util = self._format_gpu_utilization(spec)
            cost_hr, idle_day = self._compute_deployment_costs(spec, gpu_cost_override)

            cost_str = f"${cost_hr:.2f}" if cost_hr is not None else "[dim]N/A[/dim]"
            idle_str = self._format_idle_cost(idle_day, spec)

            table.add_row(
                model_name,
                spec.engine.name,
                gpu_info,
                str(spec.resources.replicas),
                gpu_util,
                cost_str,
                idle_str,
                spec.metadata.namespace,
            )

        self.console.print()
        self.console.print(table)
        self.print_waste_summary(modelspecs, gpu_cost_override)
        self.console.print()
    
    def generate_detailed_table(self, modelspec: ModelSpec) -> None:
        """
        Generate and print a detailed table for a single ModelSpec.
        
        Args:
            modelspec: ModelSpec object.
        """
        table = Table(
            title=f"ModelSpec: {modelspec.metadata.name}",
            show_header=True,
            header_style="bold",
        )
        
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        # Metadata
        table.add_row("Namespace", modelspec.metadata.namespace)
        table.add_row("Collector Version", modelspec.metadata.collector_version)
        
        # Model
        table.add_row("Model Name", modelspec.model.name or "Unknown")
        if modelspec.model.architecture:
            table.add_row("Architecture", modelspec.model.architecture)
        if modelspec.model.parameters:
            table.add_row("Parameters", modelspec.model.parameters)
        
        # Engine
        table.add_row("Engine", modelspec.engine.name)
        table.add_row(
            "Detection Confidence",
            f"{modelspec.engine.detection_confidence:.0%}",
        )
        
        # Inference
        if modelspec.inference.precision:
            table.add_row("Precision", modelspec.inference.precision)
        if modelspec.inference.tensor_parallel_size:
            table.add_row(
                "Tensor Parallel Size",
                str(modelspec.inference.tensor_parallel_size),
            )
        if modelspec.inference.max_model_len:
            table.add_row(
                "Max Model Length",
                str(modelspec.inference.max_model_len),
            )
        
        # Resources
        table.add_row("Replicas", str(modelspec.resources.replicas))
        
        self.console.print()
        self.console.print(table)
        self.console.print()
    
    def generate_framework_summary(
        self,
        modelspecs: list[ModelSpec],
    ) -> dict[str, int]:
        """
        Generate framework distribution summary.
        
        Args:
            modelspecs: List of ModelSpec objects.
            
        Returns:
            Dictionary of framework names to counts.
        """
        distribution: dict[str, int] = {}
        
        for spec in modelspecs:
            framework = spec.engine.name
            distribution[framework] = distribution.get(framework, 0) + 1
        
        return distribution
    
    def print_framework_summary(self, modelspecs: list[ModelSpec]) -> None:
        """
        Print framework distribution to console.
        
        Args:
            modelspecs: List of ModelSpec objects.
        """
        distribution = self.generate_framework_summary(modelspecs)
        
        if not distribution:
            return
        
        self.console.print("[bold]Framework Distribution:[/bold]")
        
        total = sum(distribution.values())
        for framework, count in sorted(
            distribution.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            percentage = (count / total) * 100 if total > 0 else 0
            self.console.print(
                f"  - {framework:12s}: {count:3d} deployment(s) ({percentage:.0f}%)"
            )
    
    def _estimate_gpu_cost(self, gpu_type: str, override: Optional[float]) -> float:
        """Return $/hr for a single GPU of the given type."""
        if override is not None:
            return override
        if gpu_type in _GPU_COST_PER_HOUR:
            return _GPU_COST_PER_HOUR[gpu_type]
        # Partial match (e.g. "NVIDIA-A100-SXM4-80GB" → "A100-SXM4-80GB")
        gpu_upper = gpu_type.upper()
        for key, cost in _GPU_COST_PER_HOUR.items():
            if key.upper() in gpu_upper:
                return cost
        return _DEFAULT_GPU_COST_PER_HOUR

    def _compute_deployment_costs(
        self,
        spec: ModelSpec,
        gpu_cost_override: Optional[float],
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Return (cost_per_hr, idle_cost_per_day) for the deployment.

        cost_per_hr  — total GPU spend across all replicas.
        idle_cost_per_day — GPU dollars being wasted (util < threshold).
        Either value is None when data is insufficient.
        """
        gpus = spec.resources.gpus
        if not gpus:
            return None, None

        cost_per_hr = sum(
            self._estimate_gpu_cost(gpu.type, gpu_cost_override) for gpu in gpus
        ) * spec.resources.replicas

        utilizations = [gpu.utilization for gpu in gpus if gpu.utilization is not None]
        if not utilizations:
            return cost_per_hr, None

        avg_util = sum(utilizations) / len(utilizations)
        idle_fraction = max(0.0, 1.0 - avg_util / 100.0)
        idle_cost_per_day = cost_per_hr * idle_fraction * 24.0

        return cost_per_hr, idle_cost_per_day

    def _format_idle_cost(
        self,
        idle_day: Optional[float],
        spec: ModelSpec,
    ) -> str:
        """Format the idle waste column with color coding."""
        if idle_day is None:
            gpus = spec.resources.gpus
            if not gpus:
                return "[dim]N/A[/dim]"
            return "[dim]util unknown[/dim]"

        utilizations = [
            gpu.utilization for gpu in spec.resources.gpus if gpu.utilization is not None
        ]
        avg_util = sum(utilizations) / len(utilizations) if utilizations else 100

        formatted = f"${idle_day:,.2f}"
        if avg_util < _IDLE_THRESHOLD:
            return f"[red]{formatted}[/red]"
        return f"[green]{formatted}[/green]"

    def print_waste_summary(
        self,
        modelspecs: list[ModelSpec],
        gpu_cost_override: Optional[float] = None,
    ) -> None:
        """Print a cost summary panel below the table."""
        total_cost_hr = 0.0
        total_idle_day = 0.0
        missing_util = 0

        for spec in modelspecs:
            cost_hr, idle_day = self._compute_deployment_costs(spec, gpu_cost_override)
            if cost_hr is not None:
                total_cost_hr += cost_hr
            if idle_day is not None:
                total_idle_day += idle_day
            elif spec.resources.gpus:
                missing_util += 1

        lines: list[str] = []
        if total_cost_hr > 0:
            lines.append(f"  Total GPU spend rate : [bold]${total_cost_hr:,.2f}/hr[/bold]")
        if total_idle_day > 0:
            total_idle_yr = total_idle_day * 365
            lines.append(
                f"  Estimated idle waste : [bold red]${total_idle_day:,.2f}/day[/bold red]"
                f"  ([dim]${total_idle_yr:,.0f}/yr[/dim])"
            )
        if missing_util > 0:
            lines.append(
                f"  [dim]{missing_util} deployment(s) missing GPU util data "
                f"— re-run without --no-exec for full cost breakdown[/dim]"
            )

        if lines:
            self.console.print(Panel("\n".join(lines), title="Cost Summary", expand=False))

    def _format_gpu_info(self, spec: ModelSpec) -> str:
        """Format GPU information for display."""
        gpus = spec.resources.gpus
        if not gpus:
            return "N/A"
        
        # Group by GPU type
        gpu_counts: dict[str, int] = {}
        for gpu in gpus:
            gpu_type = gpu.type
            gpu_counts[gpu_type] = gpu_counts.get(gpu_type, 0) + 1
        
        parts = []
        for gpu_type, count in gpu_counts.items():
            parts.append(f"{count}x{gpu_type}")
        
        return ", ".join(parts)
    
    def _format_gpu_utilization(self, spec: ModelSpec) -> str:
        """Format average GPU utilization for display."""
        gpus = spec.resources.gpus
        if not gpus:
            return "N/A"
        
        utilizations = [gpu.utilization for gpu in gpus if gpu.utilization is not None]
        if not utilizations:
            return "N/A"
        
        avg_util = sum(utilizations) / len(utilizations)
        
        # Color code based on utilization
        if avg_util >= 80:
            return f"[green]{avg_util:.0f}%[/green]"
        elif avg_util >= 50:
            return f"[yellow]{avg_util:.0f}%[/yellow]"
        else:
            return f"[red]{avg_util:.0f}%[/red]"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Formatted duration string.
    """
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"
