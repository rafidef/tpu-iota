"""
Statistics tracker for the miner pool dashboard.

The tracker is designed to be lightweight and safe to call from async code paths
without imposing significant locking overhead. Callers simply invoke the record
methods whenever an event occurs (activation processed, bytes transferred,
loss computed). Aggregated views are exposed via properties and helper methods
that the dashboard can query on each refresh cycle.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import monotonic
from typing import Deque, Iterable

import torch


@dataclass(slots=True)
class ActivationSample:
    """Represents a single activation processing event."""

    timestamp: float
    direction: str  # "forward" or "backward"


@dataclass(slots=True)
class LossSample:
    """Stores a scalar loss value with the time it was recorded."""

    timestamp: float
    value: float


@dataclass(slots=True)
class StatsTracker:
    """Accumulates miner runtime statistics for dashboard consumption."""

    activation_history_window: float = 300.0  # seconds
    loss_history_size: int = 50
    current_layer: int | None = None
    current_phase: str | None = None
    remote_epoch: int | None = None
    local_epoch: int | None = None
    run_id: str | None = None
    _forward_count: int = 0
    _backward_count: int = 0
    _download_bytes: int = 0
    _upload_bytes: int = 0
    _activations: Deque[ActivationSample] = field(default_factory=deque)
    _loss_history: Deque[LossSample] = field(default_factory=lambda: deque(maxlen=50))

    def __post_init__(self) -> None:
        # Ensure deque maxlen respects configuration
        if self._loss_history.maxlen != self.loss_history_size:
            self._loss_history = deque(self._loss_history, maxlen=self.loss_history_size)

    # --- Recording helpers -------------------------------------------------

    def record_forward(self, *, timestamp: float | None = None) -> None:
        """Record a forward activation being processed."""
        self._forward_count += 1
        self._record_activation(direction="forward", timestamp=timestamp)

    def record_backward(self, *, timestamp: float | None = None) -> None:
        """Record a backward activation being processed."""
        self._backward_count += 1
        self._record_activation(direction="backward", timestamp=timestamp)

    def record_download(self, byte_count: int) -> None:
        """Add bytes downloaded."""
        if byte_count < 0:
            raise ValueError("byte_count must be non-negative")
        self._download_bytes += byte_count

    def record_upload(self, byte_count: int) -> None:
        """Add bytes uploaded."""
        if byte_count < 0:
            raise ValueError("byte_count must be non-negative")
        self._upload_bytes += byte_count

    def record_loss(self, loss_value: float, *, timestamp: float | None = None) -> None:
        """Record a loss metric for last-layer miners."""
        if timestamp is None:
            timestamp = monotonic()
        self._loss_history.append(LossSample(timestamp=timestamp, value=loss_value))

    def set_layer(self, layer: int | None) -> None:
        self.current_layer = layer

    def set_phase(self, phase: object | None) -> None:
        self.current_phase = str(phase) if phase is not None else None

    def set_remote_epoch(self, epoch: int | None) -> None:
        self.remote_epoch = epoch

    def set_local_epoch(self, epoch: int | None) -> None:
        self.local_epoch = epoch

    def set_run_id(self, run_id: object | None) -> None:
        self.run_id = str(run_id) if run_id is not None else None

    def reset(self) -> None:
        """
        Reset the StatsTracker to its initial state.

        Resets all counters, state fields, and clears history collections.
        Configuration fields (activation_history_window, loss_history_size) are preserved.
        """
        # TODO: Add config settings for these?
        self.activation_history_window = 300.0
        self.loss_history_size = 50

        # Reset state fields
        self.current_layer = None
        self.current_phase = None
        self.remote_epoch = None
        self.local_epoch = None
        self.run_id = None

        # Reset counters
        self._forward_count = 0
        self._backward_count = 0
        self._download_bytes = 0
        self._upload_bytes = 0

        # Clear history collections (preserving maxlen for _loss_history)
        self._activations.clear()
        self._loss_history.clear()

    # --- Aggregated views --------------------------------------------------

    @property
    def forward_count(self) -> int:
        return self._forward_count

    @property
    def backward_count(self) -> int:
        return self._backward_count

    @property
    def total_activations(self) -> int:
        return self._forward_count + self._backward_count

    @property
    def download_bytes(self) -> int:
        return self._download_bytes

    @property
    def upload_bytes(self) -> int:
        return self._upload_bytes

    def activation_rate(self, *, window_seconds: float | None = None) -> float:
        """
        Calculate activations processed per minute over the requested window.

        Args:
            window_seconds: Override for the lookback window; defaults to the
                tracker configuration.
        """
        if window_seconds is None:
            window_seconds = self.activation_history_window

        now = monotonic()
        self._prune_activations(now, window_seconds)

        if not self._activations:
            return 0.0

        count = sum(1 for sample in self._activations if now - sample.timestamp <= window_seconds)
        if count == 0 or window_seconds <= 0:
            return 0.0

        activations_per_second = count / min(window_seconds, now - self._activations[0].timestamp + 1e-6)
        return activations_per_second * 60.0

    def loss_history(self) -> list[LossSample]:
        """Return a copy of the recorded loss history."""
        return list(self._loss_history)

    def loss_summary(self) -> dict[str, float] | None:
        """Return min/avg/max loss statistics if available."""
        if not self._loss_history:
            return None
        losses = [sample.value for sample in self._loss_history]
        count = len(losses)
        return {
            "count": float(count),
            "min": min(losses),
            "max": max(losses),
            "avg": sum(losses) / count,
            "latest": losses[-1],
        }

    # --- Internal helpers --------------------------------------------------

    def _record_activation(self, *, direction: str, timestamp: float | None) -> None:
        if timestamp is None:
            timestamp = monotonic()
        self._activations.append(ActivationSample(timestamp=timestamp, direction=direction))
        self._prune_activations(timestamp, self.activation_history_window)

    def _prune_activations(self, current_time: float, window_seconds: float) -> None:
        """Drop activation samples older than the requested window."""
        cutoff = current_time - window_seconds
        while self._activations and self._activations[0].timestamp < cutoff:
            self._activations.popleft()


def total_bytes(samples: Iterable[memoryview | bytes | bytearray]) -> int:
    """
    Utility helper to sum byte lengths from iterable payloads.

    Not yet used, but exposed for future integration when we aggregate sizes
    from multiple activation tensors or serialized uploads.
    """
    total = 0
    for sample in samples:
        total += len(sample)
    return total


def tensor_num_bytes(tensor: torch.Tensor | None) -> int:
    """Return the number of bytes occupied by a tensor."""
    if tensor is None:
        return 0
    return tensor.element_size() * tensor.nelement()
