"""
Table output generator.

Generates formatted console tables from ModelSpec data.
"""

import re
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from piqc.core.orchestrator import FragmentedNodeInfo, PendingGPUPod, UnallocatedNodeInfo
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
_STALE_AGE_DAYS = 3  # age beyond which a deployment is flagged as possibly forgotten

# GPU tier ranking: higher number = more powerful / more expensive.
# Used to detect tier misplacement (model on a GPU tier beyond what it needs).
_GPU_TIER: dict[str, int] = {
    "T4": 0,
    "L4": 1,
    "A10": 1,
    "A10G": 1,
    "V100-SXM2-16GB": 2,
    "V100-SXM2-32GB": 2,
    "V100": 2,
    "A100-SXM4-40GB": 3,
    "A100-PCIE-40GB": 3,
    "A100-SXM4-80GB": 4,
    "A100-PCIE-80GB": 4,
    "A100": 4,
    "L40S": 4,
    "H100-PCIE-80GB": 5,
    "H100-SXM4-80GB": 5,
    "H100-NVL": 5,
    "H100-SXM5-80GB": 5,
    "H100": 5,
}
_DEFAULT_GPU_TIER = 4  # A100-class as fallback

# Representative $/hr cost per tier (used to compute misplacement waste delta).
_TIER_COST: dict[int, float] = {
    0: 0.45,  # T4
    1: 0.80,  # L4 / A10
    2: 1.00,  # V100
    3: 2.25,  # A100-40GB
    4: 3.00,  # A100-80GB
    5: 4.00,  # H100
}

# Minimum GPU tier required by model size.
# Each entry: (max_param_billions, min_tier, display_name)
# A model with param_count <= threshold can run adequately on that tier.
_MIN_TIER_BY_PARAMS: list[tuple[float, int, str]] = [
    (7.0,        0, "T4"),
    (13.0,       1, "L4/A10"),
    (34.0,       3, "A100-40GB"),
    (70.0,       4, "A100-80GB"),
    (float("inf"), 5, "H100"),
]

# Theoretical peak FLOPS in TFLOPS (bf16/fp16) per GPU.
_GPU_PEAK_TFLOPS: dict[str, float] = {
    "H100-SXM5-80GB": 1979.0,
    "H100-SXM4-80GB": 989.0,
    "H100-NVL": 989.0,
    "H100-PCIE-80GB": 756.0,
    "H100": 989.0,
    "A100-SXM4-80GB": 312.0,
    "A100-SXM4-40GB": 312.0,
    "A100-PCIE-80GB": 312.0,
    "A100-PCIE-40GB": 312.0,
    "A100": 312.0,
    "L40S": 362.0,
    "A10G": 125.0,
    "A10": 125.0,
    "L4": 121.0,
    "V100-SXM2-32GB": 125.0,
    "V100-SXM2-16GB": 125.0,
    "V100": 125.0,
    "T4": 65.0,
}
_DEFAULT_GPU_PEAK_TFLOPS = 312.0  # A100 as fallback


def _parse_param_billions(model_name: str) -> Optional[float]:
    """
    Extract parameter count in billions from a model name string.

    Handles standard naming conventions:
      - 7B, 13B, 70B, 1.5B, 0.5B
      - MoE notation 8x7B (uses active expert size, not total)
    """
    if not model_name:
        return None
    # MoE: e.g. "8x7B" — use the per-expert size as effective active params
    moe = re.search(r"\d+x(\d+(?:\.\d+)?)b", model_name, re.IGNORECASE)
    if moe:
        return float(moe.group(1))
    # Standard: e.g. "70B", "7b", "1.5B"
    match = re.search(r"(\d+(?:\.\d+)?)b", model_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None


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
        unallocated_nodes: Optional[list[UnallocatedNodeInfo]] = None,
        node_cost_override: Optional[float] = None,
        fragmented_nodes: Optional[list[FragmentedNodeInfo]] = None,
        pending_gpu_pods: Optional[list[PendingGPUPod]] = None,
    ) -> None:
        """
        Generate and print a summary table of all ModelSpecs.

        Args:
            modelspecs: List of ModelSpec objects.
            gpu_cost_override: Optional $/GPU/hr override (uses built-in lookup if None).
            unallocated_nodes: Nodes with unscheduled GPU capacity.
            node_cost_override: Optional $/GPU/hr override for unallocated node GPUs.
            fragmented_nodes: Nodes with stranded GPU slots too small for any model.
            pending_gpu_pods: Pods waiting for GPU resources that cannot be scheduled.
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
        table.add_column("Deployment",  style="cyan", no_wrap=True)
        table.add_column("Engine",      style="green")
        table.add_column("GPU",         style="yellow")
        table.add_column("Replicas",    justify="right")
        table.add_column("Age",         justify="right")
        table.add_column("GPU Util",    justify="right")
        table.add_column("MFU",         justify="right")
        table.add_column("$/1K tokens", justify="right")
        table.add_column("$/hr",        justify="right")
        table.add_column("Idle $/day",  justify="right")

        any_misplaced = False
        any_unknown = False
        any_stale = False
        for spec in modelspecs:
            model_name = spec.model.name or spec.metadata.name
            if len(model_name) > 30:
                model_name = model_name[:27] + "..."

            gpu_info = self._format_gpu_info(spec)
            age_str, is_stale = self._format_age(spec.kubernetes.creation_timestamp)
            if is_stale:
                any_stale = True
            gpu_util = self._format_gpu_utilization(spec)
            cost_hr, idle_day = self._compute_deployment_costs(spec, gpu_cost_override)

            cost_str = f"${cost_hr:.2f}" if cost_hr is not None else "[dim]N/A[/dim]"
            idle_str = self._format_idle_cost(idle_day, spec)

            mfu = self._compute_mfu(spec)
            mfu_str = self._format_mfu(mfu) if mfu is not None else "[dim]N/A[/dim]"

            cpt = self._compute_cost_per_1k_tokens(spec, gpu_cost_override)
            cpt_str = f"${cpt:.4f}" if cpt is not None else "[dim]N/A[/dim]"

            is_misplaced, _, _ = self._compute_misplacement(spec, gpu_cost_override)
            if is_misplaced:
                any_misplaced = True
                gpu_info = f"{gpu_info} [yellow]⚠[/yellow]"
            else:
                param_billions = _parse_param_billions(spec.model.name or "")
                if param_billions is not None:
                    gpu_info = f"{gpu_info} [green]✓[/green]"
                else:
                    any_unknown = True
                    gpu_info = f"{gpu_info} [dim]?[/dim]"

            table.add_row(
                model_name,
                spec.engine.name,
                gpu_info,
                str(spec.resources.replicas),
                age_str,
                gpu_util,
                mfu_str,
                cpt_str,
                cost_str,
                idle_str,
            )

        self.console.print()
        self.console.print(table)
        if any_misplaced or any_unknown or any_stale:
            legend = []
            if any_misplaced:
                legend.append("[yellow]⚠[/yellow] tier larger than this model requires")
            if any_unknown:
                legend.append("[dim]?[/dim] model size unknown — fit not checked")
            if any_stale:
                legend.append(f"[yellow]Age[/yellow] running {_STALE_AGE_DAYS}+ days — confirm it's still needed")
            self.console.print(f"  [dim]{'   ·   '.join(legend)}[/dim]")
        self.print_waste_summary(modelspecs, gpu_cost_override, unallocated_nodes, node_cost_override, fragmented_nodes, pending_gpu_pods)
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
        gpu_upper = gpu_type.upper()
        for key, cost in _GPU_COST_PER_HOUR.items():
            if key.upper() in gpu_upper:
                return cost
        return _DEFAULT_GPU_COST_PER_HOUR

    def _estimate_gpu_peak_tflops(self, gpu_type: str) -> float:
        """Return theoretical peak TFLOPS (bf16) for a single GPU."""
        if gpu_type in _GPU_PEAK_TFLOPS:
            return _GPU_PEAK_TFLOPS[gpu_type]
        gpu_upper = gpu_type.upper()
        for key, tflops in _GPU_PEAK_TFLOPS.items():
            if key.upper() in gpu_upper:
                return tflops
        return _DEFAULT_GPU_PEAK_TFLOPS

    def _compute_mfu(self, spec: ModelSpec) -> Optional[float]:
        """
        Compute Model FLOPS Utilization (MFU).

        MFU = observed_FLOPS / peak_FLOPS
        observed_FLOPS = 2 × param_count × generation_tokens_per_sec
        peak_FLOPS     = sum(peak_tflops per GPU) × replicas
        """
        if not spec.runtime_state or not spec.runtime_state.vllm:
            return None
        gen_tps = spec.runtime_state.vllm.generation_tokens_per_sec
        if not gen_tps or gen_tps <= 0:
            return None

        param_billions = _parse_param_billions(spec.model.name or "")
        if param_billions is None:
            return None

        gpus = spec.resources.gpus
        if not gpus:
            return None

        observed_flops = 2.0 * param_billions * 1e9 * gen_tps
        peak_tflops = sum(
            self._estimate_gpu_peak_tflops(gpu.type) for gpu in gpus
        ) * spec.resources.replicas
        peak_flops = peak_tflops * 1e12

        return min(observed_flops / peak_flops, 1.0)

    def _compute_cost_per_1k_tokens(
        self,
        spec: ModelSpec,
        gpu_cost_override: Optional[float],
    ) -> Optional[float]:
        """
        Compute cost per 1K generated tokens.

        $/1K tokens = ($/hr / (gen_tokens_per_sec × 3600)) × 1000
        """
        if not spec.runtime_state or not spec.runtime_state.vllm:
            return None
        gen_tps = spec.runtime_state.vllm.generation_tokens_per_sec
        if not gen_tps or gen_tps <= 0:
            return None

        cost_hr, _ = self._compute_deployment_costs(spec, gpu_cost_override)
        if cost_hr is None or cost_hr <= 0:
            return None

        return (cost_hr / (gen_tps * 3600.0)) * 1000.0

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

    def _format_mfu(self, mfu: float) -> str:
        """Format MFU percentage with color coding."""
        pct = mfu * 100
        formatted = f"{pct:.1f}%"
        if pct >= 30:
            return f"[green]{formatted}[/green]"
        elif pct >= 10:
            return f"[yellow]{formatted}[/yellow]"
        return f"[red]{formatted}[/red]"

    def print_waste_summary(
        self,
        modelspecs: list[ModelSpec],
        gpu_cost_override: Optional[float] = None,
        unallocated_nodes: Optional[list[UnallocatedNodeInfo]] = None,
        node_cost_override: Optional[float] = None,
        fragmented_nodes: Optional[list[FragmentedNodeInfo]] = None,
        pending_gpu_pods: Optional[list[PendingGPUPod]] = None,
    ) -> None:
        """Print a cost summary panel below the table."""
        total_cost_hr = 0.0
        idle_day = 0.0
        missing_util = 0

        for spec in modelspecs:
            cost_hr, spec_idle_day = self._compute_deployment_costs(spec, gpu_cost_override)
            if cost_hr is not None:
                total_cost_hr += cost_hr
            if spec_idle_day is not None:
                idle_day += spec_idle_day
            elif spec.resources.gpus:
                missing_util += 1

        # Category 2: unallocated node GPUs (dark capacity)
        # Exclude fragmented nodes — they are counted separately and more precisely.
        fragmented_node_names = {n.node_name for n in (fragmented_nodes or [])}
        dark_day = 0.0
        dark_gpu_count = 0
        for node in (unallocated_nodes or []):
            if node.node_name in fragmented_node_names:
                continue
            rate = self._estimate_gpu_cost(node.gpu_type, node_cost_override or gpu_cost_override)
            dark_day += rate * node.unallocated_gpus * 24.0
            dark_gpu_count += node.unallocated_gpus

        # Category 3: tier misplacement (model on a higher GPU tier than required)
        misplaced_day = 0.0
        misplaced_count = 0
        for spec in modelspecs:
            is_misplaced, waste, _ = self._compute_misplacement(spec, gpu_cost_override)
            if is_misplaced and waste is not None:
                misplaced_day += waste
                misplaced_count += 1

        # Category 4: fragmented GPU slots (stranded, cannot fit any model)
        frag_day = 0.0
        frag_gpu_count = 0
        for node in (fragmented_nodes or []):
            rate = self._estimate_gpu_cost(node.gpu_type, node_cost_override or gpu_cost_override)
            frag_day += rate * node.stranded_gpus * 24.0
            frag_gpu_count += node.stranded_gpus

        total_leak_day = idle_day + dark_day + misplaced_day + frag_day

        lines: list[str] = []
        if total_cost_hr > 0:
            lines.append(f"  Total GPU spend rate     : [bold]${total_cost_hr:,.2f}/hr[/bold]")

        lines.append("")

        if idle_day > 0:
            lines.append(
                f"  Leased & idle (util <{_IDLE_THRESHOLD}%) : [bold red]${idle_day:,.2f}/day[/bold red]"
                f"  [dim](low utilization — may reflect traffic patterns; worth investigating)[/dim]"
            )
        if dark_day > 0:
            lines.append(
                f"  Unallocated nodes        : [bold red]${dark_day:,.2f}/day[/bold red]"
                f"  [dim]({dark_gpu_count} GPU(s) with no pods scheduled)[/dim]"
            )
        if misplaced_day > 0:
            lines.append(
                f"  Tier misplacement        : [bold red]${misplaced_day:,.2f}/day[/bold red]"
                f"  [dim]({misplaced_count} model(s) on oversized GPU tier)[/dim]"
            )
        if frag_day > 0:
            lines.append(
                f"  Fragmented nodes         : [bold red]${frag_day:,.2f}/day[/bold red]"
                f"  [dim]({frag_gpu_count} GPU(s) stranded — too few to fit any model)[/dim]"
            )

        # Pending GPU pods — active demand blocked by insufficient GPU layout
        if pending_gpu_pods:
            for pod in pending_gpu_pods:
                wait_str = (
                    f"{pod.pending_minutes:.0f} min"
                    if pod.pending_minutes < 60
                    else f"{pod.pending_minutes / 60:.1f} hr"
                )
                lines.append(
                    f"  [yellow]⚠ Pending[/yellow]  {pod.namespace}/{pod.pod_name}"
                    f"  [dim]needs {pod.gpus_needed} GPU(s) — waiting {wait_str}[/dim]"
                )

        if total_leak_day > 0:
            total_leak_yr = total_leak_day * 365
            lines.append("")
            lines.append(
                f"  Total estimated leak     : [bold red]${total_leak_day:,.2f}/day[/bold red]"
                f"  [dim](${total_leak_yr:,.0f}/yr at current rate)[/dim]"
            )
            lines.append("")
            confirmed = []
            investigate = []
            if dark_day > 0:
                confirmed.append("unallocated nodes")
            if misplaced_day > 0:
                confirmed.append("tier misplacement")
            if frag_day > 0:
                confirmed.append("fragmented capacity")
            if idle_day > 0:
                investigate.append("low GPU utilization")
            if confirmed:
                lines.append(f"  [bold]Confirmed waste[/bold]   : {', '.join(confirmed)}")
            if investigate:
                lines.append(f"  [yellow]Signals to investigate[/yellow]: {', '.join(investigate)} (verify against traffic data)")

        # Avg MFU across deployments that have runtime metrics
        mfus = [m for spec in modelspecs if (m := self._compute_mfu(spec)) is not None]
        if mfus:
            avg_mfu = sum(mfus) / len(mfus) * 100
            mfu_color = "green" if avg_mfu >= 30 else ("yellow" if avg_mfu >= 10 else "red")
            lines.append(
                f"\n  Avg MFU (active deployments) : [{mfu_color}]{avg_mfu:.1f}%[/{mfu_color}]"
                f"  [dim](healthy range: 30–60%)[/dim]"
            )

        if missing_util > 0:
            lines.append(
                f"\n  [dim]{missing_util} deployment(s) missing GPU util data"
                f" — re-run without --no-exec for full breakdown[/dim]"
            )

        if lines:
            self.console.print(Panel("\n".join(lines), title="Cost Summary", expand=False))

        self.console.print(
            "  [dim]─────────────────────────────────────────────────────────────[/dim]"
        )
        self.console.print(
            "  [bold cyan]→ Want to know what this waste is actually costing you?[/bold cyan]\n"
            "  [dim]Paralleliq turns these signals into confirmed findings with dollar impact,\n"
            "  continuous monitoring, and automated remediation — so you act on facts, not guesses.\n"
            "  Running proprietary models or on-prem hardware? We'll configure it for your exact costs.\n"
            "  Free to get started:[/dim] [bold]paralleliq.ai[/bold]"
            "  [dim]  ·  Questions?[/dim] [bold]sam@paralleliq.ai[/bold]\n"
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
    
    def _format_age(self, creation_timestamp: Optional[str]) -> tuple[str, bool]:
        """
        Format deployment age as a short duration string.

        Returns (display_string, is_stale) where is_stale flags deployments
        running longer than _STALE_AGE_DAYS — candidates for "did you forget
        to tear this down?".
        """
        if not creation_timestamp:
            return "[dim]?[/dim]", False

        try:
            created = datetime.fromisoformat(creation_timestamp)
        except ValueError:
            return "[dim]?[/dim]", False

        now = datetime.now(created.tzinfo) if created.tzinfo else datetime.now(timezone.utc).replace(tzinfo=None)
        seconds = (now - created).total_seconds()

        if seconds < 3600:
            text = f"{max(int(seconds // 60), 1)}m"
        elif seconds < 86400:
            text = f"{int(seconds // 3600)}h"
        else:
            text = f"{int(seconds // 86400)}d"

        is_stale = seconds >= _STALE_AGE_DAYS * 86400
        return (f"[yellow]{text}[/yellow]" if is_stale else text), is_stale

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

    def _get_gpu_tier(self, gpu_type: str) -> int:
        """Return the tier level for a GPU type string."""
        if gpu_type in _GPU_TIER:
            return _GPU_TIER[gpu_type]
        gpu_upper = gpu_type.upper()
        for key, tier in _GPU_TIER.items():
            if key.upper() in gpu_upper:
                return tier
        return _DEFAULT_GPU_TIER

    def _compute_misplacement(
        self,
        spec: ModelSpec,
        gpu_cost_override: Optional[float],
    ) -> tuple[bool, Optional[float], Optional[str]]:
        """
        Detect GPU tier misplacement for a deployment.

        Returns (is_misplaced, wasted_cost_per_day, min_gpu_label).
        Misplacement means the model is running on a higher GPU tier than its
        parameter count requires, wasting the cost delta per GPU per day.
        """
        gpus = spec.resources.gpus
        if not gpus:
            return False, None, None

        param_billions = _parse_param_billions(spec.model.name or "")
        if param_billions is None:
            return False, None, None

        # Determine minimum required tier for this model size
        min_tier = 5
        min_gpu_label = "H100"
        for threshold, tier, label in _MIN_TIER_BY_PARAMS:
            if param_billions <= threshold:
                min_tier = tier
                min_gpu_label = label
                break

        actual_gpu_type = gpus[0].type
        actual_tier = self._get_gpu_tier(actual_gpu_type)

        if actual_tier <= min_tier:
            return False, None, None

        # Waste = cost delta between actual tier and minimum required tier per GPU
        actual_cost = self._estimate_gpu_cost(actual_gpu_type, gpu_cost_override)
        min_cost = _TIER_COST.get(min_tier, _TIER_COST[0])
        cost_delta = actual_cost - min_cost
        if cost_delta <= 0:
            return False, None, None

        total_gpus = len(gpus) * spec.resources.replicas
        wasted_per_day = cost_delta * total_gpus * 24.0
        return True, wasted_per_day, min_gpu_label


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
