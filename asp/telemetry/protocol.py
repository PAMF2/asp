"""
JSON-RPC 2.0 Protocol Implementation

Minimal, spec-compliant JSON-RPC 2.0 message construction and parsing.
No external dependencies -- uses stdlib json only.

Spec: https://www.jsonrpc.org/specification
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any


JSONRPC_VERSION = "2.0"


@dataclass(frozen=True)
class JsonRpcRequest:
    """A JSON-RPC 2.0 request."""
    method: str
    params: dict[str, Any]
    id: str | None = None  # None = notification (no response expected)

    def to_json(self) -> str:
        obj: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "method": self.method,
            "params": self.params,
        }
        if self.id is not None:
            obj["id"] = self.id
        return json.dumps(obj, separators=(",", ":"))

    @classmethod
    def create(
        cls, method: str, params: dict[str, Any], notification: bool = False
    ) -> JsonRpcRequest:
        return cls(
            method=method,
            params=params,
            id=None if notification else str(uuid.uuid4()),
        )


@dataclass(frozen=True)
class JsonRpcResponse:
    """A JSON-RPC 2.0 response."""
    id: str
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None

    def to_json(self) -> str:
        obj: dict[str, Any] = {"jsonrpc": JSONRPC_VERSION, "id": self.id}
        if self.error is not None:
            obj["error"] = {
                "code": self.error.code,
                "message": self.error.message,
                "data": self.error.data,
            }
        else:
            obj["result"] = self.result
        return json.dumps(obj, separators=(",", ":"))


@dataclass(frozen=True)
class JsonRpcError:
    """JSON-RPC 2.0 error object."""
    code: int
    message: str
    data: dict[str, Any] | None = None


# Standard error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

# ASP custom error codes (application-defined range: -32000 to -32099)
THREAT_DETECTED = -32001
THRESHOLD_FAILED = -32002
ATTESTATION_FAILED = -32003


def parse_request(raw: str) -> JsonRpcRequest:
    """Parse a JSON-RPC 2.0 request from raw JSON string."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    if obj.get("jsonrpc") != JSONRPC_VERSION:
        raise ValueError("Not a JSON-RPC 2.0 message")

    return JsonRpcRequest(
        method=obj["method"],
        params=obj.get("params", {}),
        id=obj.get("id"),
    )


def create_batch(requests: list[JsonRpcRequest]) -> str:
    """Create a JSON-RPC 2.0 batch request."""
    return json.dumps(
        [json.loads(r.to_json()) for r in requests],
        separators=(",", ":"),
    )
