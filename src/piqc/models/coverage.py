"""
Coverage report models.

Represents what piqc could observe about a cluster's GPU infrastructure,
serving framework, and observability stack during a scan. Distinct from
ModelSpec, which describes a single detected inference deployment — a
CoverageReport describes the cluster's instrumentation as a whole.
"""

from dataclasses import dataclass, field
from typing import Literal

CoverageStatus = Literal["detected", "partial", "absent"]


@dataclass
class CoverageCheck:
    """A single coverage finding — one row in the coverage report.

    status is deliberately three-valued, not pass/fail: "absent" means
    undetected from here, not confirmed missing — RBAC scope or a
    non-standard naming convention can produce the same result as a true
    absence, and the report should not claim more certainty than the scan
    actually has.
    """

    name: str
    status: CoverageStatus
    detail: str


@dataclass
class CoverageReport:
    """Full coverage report: what's visible across three layers."""

    gpu_hardware: list[CoverageCheck] = field(default_factory=list)
    serving_framework: list[CoverageCheck] = field(default_factory=list)
    observability: list[CoverageCheck] = field(default_factory=list)

    @property
    def all_checks(self) -> list[CoverageCheck]:
        return self.gpu_hardware + self.serving_framework + self.observability

    def count_by_status(self, status: CoverageStatus) -> int:
        return sum(1 for c in self.all_checks if c.status == status)
