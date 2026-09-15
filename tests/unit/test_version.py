"""Guards against __version__/pyproject.toml drift.

Found 2026-09-15: pyproject.toml's version was bumped to 1.7.0 for the
podSpecSnapshot release, but piqc/__init__.py's __version__ constant --
which actually ships to customers as every scan's collector_version -- was
missed and stayed "1.6.0". Same class of drift the Dockerfile's own
piqc>=1.6.0 comment already documents for a fact key shipped without its
own version bump. Since a published PyPI version can never be edited or
replaced, catching this before a release (not after) is the only fix that
matters.
"""

import tomllib
from pathlib import Path

import piqc


def test_dunder_version_matches_pyproject():
    pyproject = tomllib.loads((Path(__file__).parents[2] / "pyproject.toml").read_text())
    pyproject_version = pyproject["tool"]["poetry"]["version"]
    assert piqc.__version__ == pyproject_version, (
        f"piqc.__version__ ({piqc.__version__!r}) does not match "
        f"pyproject.toml's version ({pyproject_version!r}) -- update "
        "piqc/__init__.py's __version__ before tagging a release."
    )
