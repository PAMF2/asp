"""
TelemetryEmitter -- sends defense events via JSON-RPC 2.0.

Responsible for converting internal domain objects to telemetry
payloads and dispatching them to the monitoring endpoint.

Batches events for efficiency.  Async to avoid blocking the
defense pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from asp.config import ASPConfig
from asp.telemetry.protocol import JsonRpcRequest, create_batch
from asp.telemetry.schemas import (
    MitigationPayloadTelemetry,
    ThresholdSignatureBlockPayload,
    ThreatSignatureVectorPayload,
)
from asp.types import MitigationPayload, ThresholdSignatureBlock, ThreatVector

logger = logging.getLogger(__name__)


class TelemetryEmitter:
    """Emit defense telemetry as JSON-RPC 2.0 notifications."""

    def __init__(self, config: ASPConfig) -> None:
        self._config = config
        self._buffer: list[JsonRpcRequest] = []
        self._lock = asyncio.Lock()

    async def emit_threat(self, threat: ThreatVector, request_id: str) -> None:
        """Emit a threat_signature_vector event."""
        payload = ThreatSignatureVectorPayload(
            request_id=request_id,
            embedding=threat.to_list(),
            max_attack_similarity=threat.max_attack_similarity,
            nearest_attack_id=threat.nearest_attack_id,
            threat_level=threat.threat_level.name,
            timestamp=threat.timestamp,
        )
        req = JsonRpcRequest.create(
            method="asp.threat_signature_vector",
            params=payload.to_dict(),
            notification=True,
        )
        await self._buffer_and_flush(req)

    async def emit_mitigation(
        self, mitigation: MitigationPayload, request_id: str
    ) -> None:
        """Emit a mitigation_payload event."""
        payload = MitigationPayloadTelemetry(
            request_id=request_id,
            defense_module=mitigation.defense_module,
            action=mitigation.action.name,
            explanation=mitigation.explanation,
            metadata=mitigation.sanitized_context.metadata,
        )
        req = JsonRpcRequest.create(
            method="asp.mitigation_payload",
            params=payload.to_dict(),
            notification=True,
        )
        await self._buffer_and_flush(req)

    async def emit_threshold_block(
        self, block: ThresholdSignatureBlock
    ) -> None:
        """Emit a threshold_signature_block event."""
        payload = ThresholdSignatureBlockPayload(
            request_id=block.request_id,
            verdict=block.verdict.name,
            threshold=block.threshold,
            total_nodes=block.total_nodes,
            shares_collected=len(block.shares),
            is_valid=block.is_valid,
            aggregated_signature_hex=block.aggregated_signature.hex(),
        )
        req = JsonRpcRequest.create(
            method="asp.threshold_signature_block",
            params=payload.to_dict(),
            notification=True,
        )
        await self._buffer_and_flush(req)

    async def _buffer_and_flush(self, request: JsonRpcRequest) -> None:
        """Buffer requests and flush when batch size reached."""
        async with self._lock:
            self._buffer.append(request)
            if len(self._buffer) >= self._config.telemetry_batch_size:
                await self._flush()

    async def flush(self) -> None:
        """Force flush buffered events."""
        async with self._lock:
            await self._flush()

    async def _flush(self) -> None:
        """Send buffered events to telemetry endpoint."""
        if not self._buffer:
            return

        batch = create_batch(self._buffer)
        count = len(self._buffer)
        self._buffer.clear()

        # In production, send via HTTP POST to telemetry endpoint.
        # For hackathon, log the batch.
        logger.info(
            "Telemetry batch sent (%d events) to %s",
            count,
            self._config.telemetry_endpoint,
        )
        # Placeholder for actual HTTP transport:
        # async with aiohttp.ClientSession() as session:
        #     await session.post(self._config.telemetry_endpoint, data=batch)
