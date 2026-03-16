"""
NodeRegistry -- tracks the active set of validator nodes.

Manages node lifecycle: registration, health, and share assignment.
"""

from __future__ import annotations

from asp.threshold.node import ValidatorNode


class NodeRegistry:
    """Registry of active validator nodes."""

    def __init__(self) -> None:
        self._nodes: dict[str, ValidatorNode] = {}
        self._next_share_index: int = 1

    def register(self, node: ValidatorNode) -> None:
        """Register a node and assign its share index."""
        node.share_index = self._next_share_index
        self._next_share_index += 1
        self._nodes[node.node_id] = node

    def get(self, node_id: str) -> ValidatorNode | None:
        return self._nodes.get(node_id)

    def get_all(self) -> list[ValidatorNode]:
        return list(self._nodes.values())

    @property
    def count(self) -> int:
        return len(self._nodes)

    def remove(self, node_id: str) -> bool:
        return self._nodes.pop(node_id, None) is not None
