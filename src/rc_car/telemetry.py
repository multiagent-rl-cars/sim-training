"""Inspectable rollout statistics, with no simulator or tracking-service imports."""

import math
from collections.abc import Mapping, Sequence

import torch


class RewardRollup:
    """Accumulate weighted terms before dt and their actual reward contributions.

    Isaac Lab 2.3 stores ``weight * raw_term`` in ``_step_reward``. Pass that
    matrix and the control dt to ``add`` once per environment step. No per-step
    CPU synchronization is added; finite checks happen when taking a snapshot.
    This is a rollout summary, not an episode-return or PPO-bootstrap summary.
    """

    def __init__(self, names: Sequence[str], num_agents: int) -> None:
        if isinstance(num_agents, bool) or not isinstance(num_agents, int) or num_agents < 1:
            raise ValueError("num_agents must be a positive integer")
        if len(set(names)) != len(names) or not names or any(not name for name in names):
            raise ValueError("reward names must be nonempty and unique")
        self.names = tuple(names)
        self.num_agents = num_agents
        self.steps = 0
        self.sim_seconds = 0.0
        self._weighted_sum: torch.Tensor | None = None
        self._contribution_sum: torch.Tensor | None = None

    def add(self, weighted_before_dt: torch.Tensor, dt_seconds: float) -> None:
        """Accumulate one (agent, term) matrix without retaining or mutating it."""
        if isinstance(dt_seconds, bool) or not math.isfinite(dt_seconds) or dt_seconds <= 0:
            raise ValueError("dt_seconds must be positive and finite")
        if weighted_before_dt.shape != (self.num_agents, len(self.names)):
            raise ValueError("reward matrix must have shape (num_agents, num_terms)")
        if not weighted_before_dt.is_floating_point():
            raise ValueError("reward matrix must be floating point")
        values = weighted_before_dt.detach()
        if self._weighted_sum is None:
            self._weighted_sum = torch.zeros_like(values, dtype=torch.float64)
            self._contribution_sum = torch.zeros_like(values, dtype=torch.float64)
        elif self._weighted_sum.device != values.device:
            raise ValueError("reward device must not change within a rollout")
        assert self._contribution_sum is not None
        self._weighted_sum.add_(values)
        self._contribution_sum.add_(values, alpha=dt_seconds)
        self.steps += 1
        self.sim_seconds += dt_seconds

    def snapshot(self) -> dict[str, object]:
        """Return JSON-compatible statistics without clearing the accumulator."""
        if not self.steps or self._weighted_sum is None or self._contribution_sum is None:
            raise ValueError("cannot summarize an empty rollout")
        weighted = self._weighted_sum.cpu()
        contribution = self._contribution_sum.cpu()
        if not bool(torch.isfinite(weighted).all() & torch.isfinite(contribution).all()):
            raise ValueError("reward terms contain non-finite values")
        sample_count = self.steps * self.num_agents
        return {
            "control_steps": self.steps,
            "agent_steps": sample_count,
            "sim_seconds_per_agent": self.sim_seconds,
            "weighted_mean_before_dt": {
                name: float(weighted[:, i].sum() / sample_count)
                for i, name in enumerate(self.names)
            },
            "contribution_mean_per_agent_step": {
                name: float(contribution[:, i].sum() / sample_count)
                for i, name in enumerate(self.names)
            },
            "contribution_sum_per_agent": {
                name: contribution[:, i].tolist() for i, name in enumerate(self.names)
            },
        }

    def clear(self) -> None:
        """Start another rollout without changing the term/agent contract."""
        self.steps = 0
        self.sim_seconds = 0.0
        self._weighted_sum = None
        self._contribution_sum = None


def count_termination_causes(
    term_masks: Mapping[str, torch.Tensor], dones: torch.Tensor
) -> dict[str, int]:
    """Count each active cause on completed transitions; causes may overlap.

    A collision and off-track termination can describe the same episode. Their
    counts must not be summed and asserted equal to the number of completed
    episodes. Mask stale manager flags by the current done vector.
    """
    if dones.ndim != 1 or dones.dtype != torch.bool:
        raise ValueError("dones must be a one-dimensional boolean tensor")
    counts: dict[str, int] = {}
    for name, mask in term_masks.items():
        if mask.shape != dones.shape or mask.dtype != torch.bool or mask.device != dones.device:
            raise ValueError("termination masks must match dones shape, dtype, and device")
        counts[name] = int((mask & dones).sum().item())
    return counts
