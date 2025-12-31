"""
Metrics computation utilities for EAB experiments.
"""

import time
import psutil
import torch
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict


@dataclass
class MetricsSnapshot:
    """Single measurement snapshot."""
    token_steps: int = 0
    wall_clock_time: float = 0.0
    memory_peak_mb: float = 0.0
    tokens_per_sample: float = 0.0
    num_samples: int = 0

    # Branching-specific (EAB only)
    branch_count: int = 0
    branch_positions: List[int] = None
    final_path_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsTracker:
    """
    Tracks computational metrics during generation.

    Usage:
        tracker = MetricsTracker()
        tracker.start()

        # ... run generation ...

        tracker.record_token_steps(total_steps)
        metrics = tracker.stop()
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.start_time: Optional[float] = None
        self.start_memory: float = 0.0
        self.peak_memory: float = 0.0

        # Collected data
        self.token_steps: int = 0
        self.num_samples: int = 0
        self.branch_count: int = 0
        self.branch_positions: List[int] = []
        self.final_path_count: int = 0

    def start(self):
        """Start tracking metrics."""
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        self.peak_memory = self.start_memory

    def update_memory(self):
        """Update peak memory usage."""
        current_memory = self._get_memory_usage()
        self.peak_memory = max(self.peak_memory, current_memory)

    def record_token_steps(self, steps: int):
        """Record token processing steps."""
        self.token_steps += steps

    def record_branch(self, position: int):
        """Record a branching event."""
        self.branch_count += 1
        self.branch_positions.append(position)

    def record_samples(self, count: int):
        """Record number of samples generated."""
        self.num_samples = count

    def record_final_paths(self, count: int):
        """Record final number of paths."""
        self.final_path_count = count

    def stop(self) -> MetricsSnapshot:
        """Stop tracking and return metrics snapshot."""
        if self.start_time is None:
            raise RuntimeError("Tracker not started. Call start() first.")

        wall_clock_time = time.time() - self.start_time
        memory_peak_mb = self.peak_memory - self.start_memory

        # Compute derived metrics
        tokens_per_sample = (
            self.token_steps / self.num_samples
            if self.num_samples > 0 else 0.0
        )

        return MetricsSnapshot(
            token_steps=self.token_steps,
            wall_clock_time=wall_clock_time,
            memory_peak_mb=memory_peak_mb,
            tokens_per_sample=tokens_per_sample,
            num_samples=self.num_samples,
            branch_count=self.branch_count,
            branch_positions=self.branch_positions.copy() if self.branch_positions else [],
            final_path_count=self.final_path_count
        )

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if self.device == "cuda" and torch.cuda.is_available():
            # GPU memory
            return torch.cuda.memory_allocated() / 1024 / 1024  # Convert to MB
        else:
            # CPU memory
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB


def compute_token_steps(
    prompt_length: int,
    generated_length: int,
    num_samples: int,
    method: str = "naive"
) -> int:
    """
    Compute total token processing steps.

    Args:
        prompt_length: Length of prompt in tokens
        generated_length: Average length of generated text in tokens
        num_samples: Number of samples generated
        method: "naive" or "eab"

    Returns:
        Total token-steps (forward passes × tokens)
    """
    if method == "naive":
        # Naive: Each sample processes entire sequence independently
        total_length = prompt_length + generated_length
        return num_samples * total_length
    else:
        # EAB: More complex, should be measured directly
        # This is just an approximation
        raise NotImplementedError(
            "EAB token steps should be measured directly, not approximated"
        )


def compute_speedup(
    naive_metrics: MetricsSnapshot,
    eab_metrics: MetricsSnapshot,
    metric: str = "token_steps"
) -> float:
    """
    Compute speedup factor: Naive / EAB.

    Args:
        naive_metrics: Metrics from naive sampling
        eab_metrics: Metrics from EAB
        metric: Which metric to use for speedup

    Returns:
        Speedup factor (higher is better)
    """
    naive_value = getattr(naive_metrics, metric)
    eab_value = getattr(eab_metrics, metric)

    if eab_value == 0:
        return float('inf')

    return naive_value / eab_value


def compute_efficiency_metrics(
    naive_metrics: MetricsSnapshot,
    eab_metrics: MetricsSnapshot
) -> Dict[str, float]:
    """
    Compute all efficiency metrics.

    Returns:
        Dictionary with speedup factors for all metrics
    """
    metrics = {}

    # Speedup factors (higher is better)
    metrics['speedup_token_steps'] = compute_speedup(
        naive_metrics, eab_metrics, 'token_steps'
    )
    metrics['speedup_time'] = compute_speedup(
        naive_metrics, eab_metrics, 'wall_clock_time'
    )
    metrics['speedup_memory'] = compute_speedup(
        naive_metrics, eab_metrics, 'memory_peak_mb'
    )

    # Cost ratios (lower is better)
    metrics['cost_ratio_token_steps'] = 1.0 / metrics['speedup_token_steps']
    metrics['cost_ratio_time'] = 1.0 / metrics['speedup_time']
    metrics['cost_ratio_memory'] = 1.0 / metrics['speedup_memory']

    # Efficiency score (tokens per unit cost)
    metrics['tokens_per_second_naive'] = (
        naive_metrics.token_steps / naive_metrics.wall_clock_time
        if naive_metrics.wall_clock_time > 0 else 0.0
    )
    metrics['tokens_per_second_eab'] = (
        eab_metrics.token_steps / eab_metrics.wall_clock_time
        if eab_metrics.wall_clock_time > 0 else 0.0
    )

    return metrics


def compute_branching_stats(metrics: MetricsSnapshot) -> Dict[str, Any]:
    """
    Compute statistics about branching behavior.

    Args:
        metrics: EAB metrics snapshot

    Returns:
        Dictionary with branching statistics
    """
    stats = {
        'total_branches': metrics.branch_count,
        'final_path_count': metrics.final_path_count,
        'branch_positions': metrics.branch_positions,
    }

    if metrics.branch_positions:
        stats['avg_branch_position'] = sum(metrics.branch_positions) / len(metrics.branch_positions)
        stats['first_branch_position'] = min(metrics.branch_positions)
        stats['last_branch_position'] = max(metrics.branch_positions)
    else:
        stats['avg_branch_position'] = None
        stats['first_branch_position'] = None
        stats['last_branch_position'] = None

    return stats
