"""
CLI commands for ModelSpec Introspector.

Provides the command-line interface for scanning Kubernetes clusters
and generating ModelSpec documentation.
"""

import sys
from typing import Optional

import click
from rich.console import Console

from piqc import __version__
from piqc.core.k8s_client import K8sClient
from piqc.core.orchestrator import ScanOrchestrator
from piqc.generators.json_generator import JSONGenerator
from piqc.generators.piqc_generator import PIQCGenerator
from piqc.generators.table_generator import TableGenerator, format_duration
from piqc.generators.yaml_generator import YAMLGenerator
from piqc.utils.exceptions import (
    KubernetesConnectionError,
    RBACPermissionError,
)
from piqc.utils.logger import setup_logging, get_logger


# Use a wide fixed width when running without a TTY (e.g. kubectl logs, CI)
console = Console() if sys.stdout.isatty() else Console(width=220, force_terminal=True)
logger = get_logger(__name__)


def print_header() -> None:
    """Print application header."""
    console.print(f"ModelSpec Introspector v{__version__}")
    console.print("=" * 40)
    console.print()


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[INFO] {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[WARN] {message}", style="yellow")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[ERROR] {message}", style="red")


@click.group()
@click.version_option(version=__version__, prog_name="piqc")
def main() -> None:
    """
    ModelSpec Introspector - Kubernetes vLLM Model Discovery Tool.
    
    Discovers vLLM inference deployments in Kubernetes clusters
    and generates standardized ModelSpec documentation.
    """
    pass


@main.command()
@click.option(
    "--kubeconfig",
    type=click.Path(exists=True),
    default=None,
    help="Path to kubeconfig file. Default: ~/.kube/config",
)
@click.option(
    "--context",
    type=str,
    default=None,
    help="Kubernetes context to use. Default: current context",
)
@click.option(
    "--namespace",
    "-n",
    type=str,
    default=None,
    help="Specific namespace to scan. Default: all namespaces",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["yaml", "json", "table"]),
    default="table",
    help="Output format. Default: table",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="./output",
    help="Output directory for generated files. Default: ./output",
)
@click.option(
    "--timeout",
    type=int,
    default=30,
    help="Operation timeout in seconds. Default: 30",
)
@click.option(
    "--no-exec",
    is_flag=True,
    default=False,
    help="Disable pod exec (skip GPU metrics collection)",
)
@click.option(
    "--no-logs",
    is_flag=True,
    default=False,
    help="Disable log reading",
)
@click.option(
    "--workers",
    type=int,
    default=5,
    help="Number of parallel workers. Default: 5",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable verbose output",
)
@click.option(
    "--debug",
    is_flag=True,
    default=False,
    help="Enable debug mode with detailed trace",
)
@click.option(
    "--combined",
    is_flag=True,
    default=False,
    help="Generate a single combined output file instead of per-deployment files",
)
@click.option(
    "--collect-runtime/--no-collect-runtime",
    default=True,
    help="Collect runtime metrics via vLLM API. Default: enabled",
)
@click.option(
    "--aggregate/--no-aggregate",
    default=True,
    help="Aggregate metrics across pod replicas. Default: aggregate",
)
@click.option(
    "--mode",
    type=click.Choice(["auto", "remote", "incluster", "dry-run"]),
    default="auto",
    help="Execution mode. Default: auto-detect",
)
@click.option(
    "--output-piqc",
    is_flag=True,
    default=False,
    help="Generate piqc-facts.json output (PIQC scan v0.1 schema)",
)
@click.option(
    "--gpu-cost",
    type=float,
    default=None,
    metavar="DOLLARS",
    help="Override GPU cost in $/GPU/hr (e.g. 3.50). Default: auto-detect by GPU type.",
)
@click.option(
    "--node-cost",
    type=float,
    default=None,
    metavar="DOLLARS",
    help="Override cost in $/GPU/hr for unallocated nodes. Defaults to --gpu-cost if set.",
)
@click.option(
    "--contribute-benchmarks",
    is_flag=True,
    default=False,
    help="Contribute anonymized GPU/model performance data to the ParallelIQ benchmark dataset.",
)
def scan(
    kubeconfig: Optional[str],
    context: Optional[str],
    namespace: Optional[str],
    output_format: str,
    output: str,
    timeout: int,
    no_exec: bool,
    no_logs: bool,
    workers: int,
    verbose: bool,
    debug: bool,
    combined: bool,
    collect_runtime: bool,
    aggregate: bool,
    mode: str,
    output_piqc: bool,
    gpu_cost: Optional[float],
    node_cost: Optional[float],
    contribute_benchmarks: bool,
) -> None:
    """
    Scan Kubernetes cluster for vLLM model deployments.
    
    Discovers vLLM inference workloads and generates standardized
    ModelSpec documentation with optional runtime metrics.
    
    \b
    Examples:
        # Scan entire cluster
        piqc scan
        
        # Scan specific namespace
        piqc scan -n production
        
        # Collect runtime metrics via vLLM API
        piqc scan --collect-runtime
        
        # JSON output without aggregation
        piqc scan --format json --no-aggregate
        
        # Table output (no files generated)
        piqc scan --format table
        
        # Skip GPU metrics (faster)
        piqc scan --no-exec
    """
    # Setup logging
    setup_logging(verbose=verbose, debug=debug)
    
    # Print header
    print_header()
    
    # Connect to cluster
    print_info("Connecting to cluster...")
    
    try:
        k8s_client = K8sClient(
            kubeconfig_path=kubeconfig,
            context=context,
            timeout=timeout,
        )
        k8s_client.test_connection()
        
        conn_info = k8s_client.get_connection_info()
        console.print(f"       Context: {conn_info['context']}")
        if conn_info['cluster'] != "unknown":
            console.print(f"       Cluster: {conn_info['cluster']}")
        console.print()
        
    except KubernetesConnectionError as e:
        print_error(str(e))
        if e.details:
            console.print(f"       {e.details}")
        sys.exit(1)
    except RBACPermissionError as e:
        print_error(str(e))
        console.print("       See RBAC documentation for required permissions.")
        sys.exit(1)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if debug:
            console.print_exception()
        sys.exit(1)
    
    # Initialize orchestrator
    orchestrator = ScanOrchestrator(
        k8s_client=k8s_client,
        enable_exec=not no_exec,
        enable_logs=not no_logs,
        enable_runtime_collection=collect_runtime,
        workers=workers,
        timeout=timeout,
    )
    
    # Log mode if not auto
    if mode != "auto":
        print_info(f"Execution mode: {mode}")
    
    # Execute scan
    print_info("Scanning namespaces...")
    
    namespaces = [namespace] if namespace else None
    
    try:
        result = orchestrator.scan(namespaces=namespaces)
    except Exception as e:
        print_error(f"Scan failed: {e}")
        if debug:
            console.print_exception()
        sys.exit(1)
    
    # Print scan summary
    console.print(f"       Discovered: {result.namespaces_scanned} namespace(s)")
    console.print()
    
    print_info("Detecting inference workloads...")
    console.print(f"       Pods analyzed: {result.pods_analyzed}")
    console.print(f"       Inference deployments found: {result.deployments_found}")
    console.print()
    
    # Print framework distribution
    if result.modelspecs:
        table_gen = TableGenerator(console)
        table_gen.print_framework_summary(result.modelspecs)
        console.print()
    
    # Print warnings
    if result.warnings:
        print_info("Warnings:")
        for warning in result.warnings[:10]:  # Limit to first 10
            console.print(f"       - {warning}", style="yellow")
        if len(result.warnings) > 10:
            console.print(f"       ... and {len(result.warnings) - 10} more")
        console.print()
    
    # Generate output
    if result.modelspecs:
        # Show hint for unknown-runtime deployments
        unknown = [s for s in result.modelspecs if s.engine.name == "unknown"]
        if unknown:
            console.print(
                f"[yellow]  {len(unknown)} unknown-runtime pod(s) detected "
                f"(TGI / Triton / other). Run with --verbose to inspect.[/yellow]"
            )
            console.print()

        if output_format == "table":
            # Table output to console
            table_gen = TableGenerator(console)
            table_gen.generate_summary_table(
                result.modelspecs,
                gpu_cost_override=gpu_cost,
                unallocated_nodes=result.unallocated_nodes,
                node_cost_override=node_cost,
            )
        
        elif output_format == "yaml":
            yaml_gen = YAMLGenerator()
            if combined:
                output_file = f"{output}/modelspecs.yaml"
                yaml_gen.generate_combined(result.modelspecs, output_file)
                print_info(f"Output generated:")
                console.print(f"       File: {output_file}")
            else:
                files = yaml_gen.generate_multi(result.modelspecs, output)
                print_info(f"Output generated:")
                console.print(f"       Directory: {output}")
                console.print(f"       Files: {len(files)} ModelSpec file(s)")
        
        elif output_format == "json":
            json_gen = JSONGenerator()
            if combined:
                output_file = f"{output}/modelspecs.json"
                json_gen.generate_combined(result.modelspecs, output_file)
                print_info(f"Output generated:")
                console.print(f"       File: {output_file}")
            else:
                files = json_gen.generate_multi(result.modelspecs, output)
                print_info(f"Output generated:")
                console.print(f"       Directory: {output}")
                console.print(f"       Files: {len(files)} ModelSpec file(s)")
        
        # Generate PIQC facts bundle if requested
        if output_piqc:
            piqc_gen = PIQCGenerator()
            piqc_file = piqc_gen.generate(
                modelspecs=result.modelspecs,
                output_path=output,
                cluster_context=conn_info.get('context'),
                cluster_name=conn_info.get('cluster'),
                namespaces=namespaces,
            )
            print_info(f"PIQC facts bundle generated:")
            console.print(f"       File: {piqc_file}")
        
        console.print()
    else:
        print_info("No inference deployments found.")
        console.print()
    
    # Contribute anonymized benchmarks if requested
    if contribute_benchmarks and result.modelspecs:
        from piqc import __version__
        from piqc.telemetry import contribute
        sent, records = contribute(result.modelspecs, __version__)
        if sent > 0:
            console.print(f"[dim]  Contributed {sent} anonymized benchmark record(s) to paralleliq.ai[/dim]")
            if verbose:
                import json
                console.print("[dim]  Payload (no identifying info):[/dim]")
                for r in records:
                    console.print(f"[dim]    {json.dumps(r)}[/dim]")
        else:
            console.print("[dim]  Benchmark contribution failed (non-fatal) — check connectivity[/dim]")
        console.print()

    # Print timing
    print_info(f"Scan completed in {format_duration(result.duration_seconds)}")
    
    # Exit with appropriate code
    if result.errors:
        sys.exit(1)


@main.command()
@click.option(
    "--kubeconfig",
    type=click.Path(exists=True),
    default=None,
    help="Path to kubeconfig file",
)
@click.option(
    "--context",
    type=str,
    default=None,
    help="Kubernetes context to use",
)
def test_connection(
    kubeconfig: Optional[str],
    context: Optional[str],
) -> None:
    """
    Test connection to Kubernetes cluster.
    
    Verifies that the tool can connect to the cluster and has
    necessary permissions for scanning.
    """
    print_header()
    print_info("Testing cluster connection...")
    
    try:
        k8s_client = K8sClient(
            kubeconfig_path=kubeconfig,
            context=context,
        )
        k8s_client.test_connection()
        
        conn_info = k8s_client.get_connection_info()
        
        console.print()
        console.print("[green]Connection successful[/green]")
        console.print()
        console.print(f"Context: {conn_info['context']}")
        console.print(f"Cluster: {conn_info['cluster']}")
        
        # Test namespace listing
        print_info("Testing namespace access...")
        namespaces = k8s_client.list_namespaces()
        console.print(f"       Accessible namespaces: {len(namespaces)}")
        
        console.print()
        console.print("[green]All checks passed[/green]")
        
    except KubernetesConnectionError as e:
        print_error(str(e))
        if e.details:
            console.print(f"       {e.details}")
        sys.exit(1)
    except RBACPermissionError as e:
        print_error(str(e))
        sys.exit(1)
    except Exception as e:
        print_error(f"Connection test failed: {e}")
        sys.exit(1)


@main.command()
def version() -> None:
    """Display version information."""
    console.print(f"ModelSpec Introspector v{__version__}")


if __name__ == "__main__":
    main()
