"""Unit tests for post-scan lead capture (should_prompt gating and submit)."""

from unittest.mock import MagicMock, patch

from piqc.core.orchestrator import ScanResult
from piqc.leadcapture import should_prompt, submit


def _result_with_modelspecs():
    result = ScanResult()
    result.modelspecs = [MagicMock()]
    result.cloud_provider = "aws"
    return result


def test_should_prompt_requires_table_format():
    result = _result_with_modelspecs()
    with patch("sys.stdin.isatty", return_value=True), patch("sys.stdout.isatty", return_value=True):
        assert should_prompt(result, "json") is False
        assert should_prompt(result, "yaml") is False
        assert should_prompt(result, "table") is True


def test_should_prompt_requires_modelspecs():
    result = ScanResult()
    with patch("sys.stdin.isatty", return_value=True), patch("sys.stdout.isatty", return_value=True):
        assert should_prompt(result, "table") is False


def test_should_prompt_requires_interactive_terminal():
    result = _result_with_modelspecs()
    with patch("sys.stdin.isatty", return_value=False), patch("sys.stdout.isatty", return_value=True):
        assert should_prompt(result, "table") is False
    with patch("sys.stdin.isatty", return_value=True), patch("sys.stdout.isatty", return_value=False):
        assert should_prompt(result, "table") is False


def test_submit_returns_true_on_2xx():
    result = _result_with_modelspecs()
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.__enter__.return_value = mock_resp
    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert submit("Jane Doe", "Acme", "jane@acme.com", result, "1.3.0") is True


def test_submit_returns_false_on_network_error():
    result = _result_with_modelspecs()
    with patch("urllib.request.urlopen", side_effect=OSError("no network")):
        assert submit("Jane Doe", "Acme", "jane@acme.com", result, "1.3.0") is False


def test_submit_never_raises_on_failure():
    result = _result_with_modelspecs()
    with patch("urllib.request.urlopen", side_effect=Exception("boom")):
        # Should not raise
        assert submit("Jane Doe", "Acme", "jane@acme.com", result, "1.3.0") is False
