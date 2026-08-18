"""
Post-scan lead capture for PIQC.

After a scan finds inference deployments, the CLI offers a one-time,
skippable prompt: "Save & Get Custom Analysis" or "Skip for Now". If the
user opts in, we collect name/company/email and send it — along with the
cloud provider (inferred from the cluster, not asked for) and a compact
scan summary — to Paralleliq via a Web3Forms webhook.

Never runs for non-interactive invocations (piped output, CI, scripted
--format json/yaml usage) — see should_prompt().
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from urllib.parse import urlencode

from piqc.core.orchestrator import ScanResult
from piqc.utils.logger import get_logger

logger = get_logger(__name__)

# Dedicated Web3Forms form ("piqc scan leads"), routed to info@paralleliq.ai.
_WEB3FORMS_ACCESS_KEY = "7f5df0a2-ca71-4a79-9920-9554ab0f5643"

_WEB3FORMS_ENDPOINT = "https://api.web3forms.com/submit"
_TIMEOUT_SECONDS = 5


def should_prompt(result: ScanResult, output_format: str) -> bool:
    """Only prompt for a real interactive terminal session, human-facing
    output, and a scan that actually found something worth following up
    on — never for scripted/CI invocations, which must not block on stdin."""
    if output_format != "table":
        return False
    if not result.modelspecs:
        return False
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return False
    return True


def _summarize(result: ScanResult) -> str:
    unallocated_gpus = sum(n.unallocated_gpus for n in result.unallocated_nodes)
    fragmented_gpus = sum(n.stranded_gpus for n in result.fragmented_nodes)
    lines = [
        f"Cloud provider: {result.cloud_provider or 'unknown'}",
        f"Namespaces scanned: {result.namespaces_scanned}",
        f"Inference deployments found: {result.deployments_found}",
        f"Dark/unallocated GPUs: {unallocated_gpus}",
        f"Stranded (fragmented) GPUs: {fragmented_gpus}",
        f"Pending GPU pods: {len(result.pending_gpu_pods)}",
    ]
    return "\n".join(lines)


def submit(name: str, company: str, email: str, result: ScanResult, piqc_version: str) -> bool:
    """POST the captured lead + scan summary to Web3Forms. Never raises —
    a failed submission is not worth blocking or scaring the user over."""
    payload = {
        "access_key": _WEB3FORMS_ACCESS_KEY,
        "subject": f"New piqc scan lead — {company}",
        "from_name": "piqc scanner",
        "name": name,
        "company": company,
        "email": email,
        "message": _summarize(result),
        "piqc_version": piqc_version,
    }
    body = urlencode(payload).encode("utf-8")
    try:
        req = urllib.request.Request(
            _WEB3FORMS_ENDPOINT,
            data=body,
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": f"piqc/{piqc_version}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
            return 200 <= resp.status < 300
    except Exception as exc:
        logger.debug("Lead capture submit failed (non-fatal): %s", exc)
        return False


def prompt_and_submit(result: ScanResult, console, piqc_version: str) -> None:
    """Show the opt-in prompt and, if accepted, collect contact info and
    submit it. Safe to call unconditionally — checks should_prompt-style
    conditions are the caller's responsibility via should_prompt()."""
    from rich.prompt import Confirm, Prompt

    console.print()
    save = Confirm.ask(
        "[bold]Save these results and get a custom analysis from Paralleliq?[/bold]",
        default=False,
    )
    if not save:
        return

    name = Prompt.ask("  Name")
    company = Prompt.ask("  Company")
    email = Prompt.ask("  Work email")

    if submit(name, company, email, result, piqc_version):
        console.print("[dim]  Sent — someone from Paralleliq will follow up shortly.[/dim]")
    else:
        console.print(
            "[dim]  Couldn't reach Paralleliq right now (non-fatal) — "
            "email info@paralleliq.ai directly if you'd like a follow-up.[/dim]"
        )
