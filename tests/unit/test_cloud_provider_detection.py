"""Unit tests for cloud provider detection from node.spec.providerID."""

from unittest.mock import MagicMock

from piqc.core.orchestrator import detect_cloud_provider


def _node(provider_id):
    node = MagicMock()
    node.spec = MagicMock()
    node.spec.provider_id = provider_id
    return node


def test_detects_aws():
    assert detect_cloud_provider([_node("aws:///us-east-1a/i-0abcd1234")]) == "aws"


def test_detects_gcp():
    assert detect_cloud_provider([_node("gce://my-project/us-central1-a/instance-1")]) == "gcp"


def test_detects_azure():
    assert detect_cloud_provider([_node("azure:///subscriptions/x/resourceGroups/y/vm/z")]) == "azure"


def test_unknown_prefix_reported_as_is():
    assert detect_cloud_provider([_node("metal3://foo/bar")]) == "metal3"


def test_no_provider_id_is_on_prem():
    assert detect_cloud_provider([_node(None)]) == "on-prem / other"


def test_empty_node_list_is_on_prem():
    assert detect_cloud_provider([]) == "on-prem / other"


def test_uses_first_node_with_a_provider_id():
    nodes = [_node(None), _node("aws:///us-east-1a/i-0abcd1234"), _node("gce://p/z/i")]
    assert detect_cloud_provider(nodes) == "aws"


def test_node_with_no_spec_is_skipped():
    node = MagicMock()
    node.spec = None
    assert detect_cloud_provider([node]) == "on-prem / other"
