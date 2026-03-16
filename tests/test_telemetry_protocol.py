"""Tests for JSON-RPC 2.0 telemetry protocol."""

from __future__ import annotations

import json

from asp.telemetry.protocol import (
    JSONRPC_VERSION,
    JsonRpcRequest,
    create_batch,
    parse_request,
)


class TestJsonRpcProtocol:

    def test_request_serialization(self):
        req = JsonRpcRequest.create(
            method="asp.threat_signature_vector",
            params={"request_id": "123", "threat_level": "BLOCK"},
        )
        raw = req.to_json()
        parsed = json.loads(raw)

        assert parsed["jsonrpc"] == JSONRPC_VERSION
        assert parsed["method"] == "asp.threat_signature_vector"
        assert parsed["params"]["threat_level"] == "BLOCK"
        assert "id" in parsed

    def test_notification_has_no_id(self):
        req = JsonRpcRequest.create(
            method="asp.event",
            params={"data": "test"},
            notification=True,
        )
        raw = req.to_json()
        parsed = json.loads(raw)
        assert "id" not in parsed

    def test_batch_creation(self):
        requests = [
            JsonRpcRequest.create("m1", {"a": 1}),
            JsonRpcRequest.create("m2", {"b": 2}),
        ]
        batch = create_batch(requests)
        parsed = json.loads(batch)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

    def test_parse_request(self):
        raw = json.dumps({
            "jsonrpc": "2.0",
            "method": "test",
            "params": {"key": "value"},
            "id": "abc",
        })
        req = parse_request(raw)
        assert req.method == "test"
        assert req.params["key"] == "value"
        assert req.id == "abc"

    def test_parse_invalid_version_raises(self):
        import pytest

        raw = json.dumps({"jsonrpc": "1.0", "method": "test"})
        with pytest.raises(ValueError, match="Not a JSON-RPC 2.0"):
            parse_request(raw)
