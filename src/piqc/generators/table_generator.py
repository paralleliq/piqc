"""
Table output generator.

Generates formatted console tables from ModelSpec data.
"""

from typing import Optional

from rich.console import Console
from rich.table import Table

from piqc.models.modelspec import ModelSpec
from piqc.utils.logger import get_logger


logger = get_logger(__name__)


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
    
    def generate_summary_table(self, modelspecs: list[ModelSpec]) -> None:
        """
        Generate and print a summary table of all ModelSpecs.
        
        Args:
            modelspecs: List of ModelSpec objects.
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
        table.add_column("Model Name", style="cyan", no_wrap=True)
        table.add_column("Engine", style="green")
        table.add_column("GPU Type", style="yellow")
        table.add_column("Replicas", justify="right")
        table.add_column("GPU Util", justify="right")
        table.add_column("Namespace", style="dim")
        
        for spec in modelspecs:
            # Format model name
            model_name = spec.model.name or spec.metadata.name
            if len(model_name) > 25:
                model_name = model_name[:22] + "..."
            
            # Format GPU info
            gpu_info = self._format_gpu_info(spec)
            gpu_util = self._format_gpu_utilization(spec)
            
            table.add_row(
                model_name,
                spec.engine.name,
                gpu_info,
                str(spec.resources.replicas),
                gpu_util,
                spec.metadata.namespace,
            )
        
        self.console.print()
        self.console.print(table)
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
