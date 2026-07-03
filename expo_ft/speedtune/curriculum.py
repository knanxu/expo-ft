"""Speed-bin curriculum state for fixed-time and whole-chunk SpeedTune."""

from __future__ import annotations


class SpeedCurriculum:
    """Monotonically unlock an ordered speed grid from slow to fast."""

    def __init__(
        self,
        *,
        speed_grid,
        window_size: int = 20,
        success_threshold: float = 0.7,
    ):
        self.speed_grid = tuple(float(value) for value in speed_grid)
        if not self.speed_grid:
            raise ValueError("speed_grid must not be empty")
        self.window_size = int(window_size)
        self.success_threshold = float(success_threshold)
        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError("success_threshold must be in [0, 1]")

        self.max_unlocked_idx = 0
        self.recent_results = []
        self.action_counts = [0 for _ in self.speed_grid]
        self.unlock_events = []

    @property
    def max_unlocked_speed(self) -> float:
        return self.speed_grid[self.max_unlocked_idx]

    @property
    def action_limits(self) -> tuple[int, ...]:
        return (self.max_unlocked_idx,)

    @property
    def window_success_rate(self) -> float:
        if not self.recent_results:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)

    def record_action(self, action_idxs) -> None:
        indices = tuple(int(value) for value in action_idxs)
        if len(indices) != 1:
            raise ValueError("speed curriculum supports exactly one action head")
        index = indices[0]
        if not 0 <= index <= self.max_unlocked_idx:
            raise ValueError(
                f"action index {index} exceeds unlocked index {self.max_unlocked_idx}"
            )
        self.action_counts[index] += 1

    def record_episode(self, reward_success, *, episode: int, decision: int) -> bool:
        self.recent_results.append(bool(reward_success))
        if len(self.recent_results) > self.window_size:
            self.recent_results.pop(0)
        if len(self.recent_results) < self.window_size:
            return False
        if self.max_unlocked_idx >= len(self.speed_grid) - 1:
            return False
        if self.window_success_rate + 1e-12 < self.success_threshold:
            return False

        self.max_unlocked_idx += 1
        self.unlock_events.append(
            {
                "episode": int(episode),
                "decision": int(decision),
                "unlocked_idx": self.max_unlocked_idx,
                "speed": self.max_unlocked_speed,
            }
        )
        self.recent_results = []
        return True

    def state_dict(self) -> dict:
        return {
            "speed_grid": list(self.speed_grid),
            "window_size": self.window_size,
            "success_threshold": self.success_threshold,
            "max_unlocked_idx": self.max_unlocked_idx,
            "recent_results": list(self.recent_results),
            "action_counts": list(self.action_counts),
            "unlock_events": [dict(event) for event in self.unlock_events],
        }

    @classmethod
    def from_state_dict(cls, state) -> "SpeedCurriculum":
        curriculum = cls(
            speed_grid=state["speed_grid"],
            window_size=state["window_size"],
            success_threshold=state["success_threshold"],
        )
        curriculum.max_unlocked_idx = int(state["max_unlocked_idx"])
        curriculum.recent_results = [bool(value) for value in state["recent_results"]]
        curriculum.action_counts = [int(value) for value in state["action_counts"]]
        curriculum.unlock_events = [dict(event) for event in state["unlock_events"]]
        if not 0 <= curriculum.max_unlocked_idx < len(curriculum.speed_grid):
            raise ValueError("invalid max_unlocked_idx in curriculum state")
        if len(curriculum.action_counts) != len(curriculum.speed_grid):
            raise ValueError("action_counts length does not match speed_grid")
        if len(curriculum.recent_results) > curriculum.window_size:
            raise ValueError("recent_results exceeds curriculum window_size")
        return curriculum


def build_speed_curriculum(config, backend):
    """Build curriculum for the two ordered, single-speed SpeedTune backends."""
    enabled = bool(config.get("curriculum_enabled", True))
    if not enabled or backend.name not in ("fixed_time", "chunk_toppra"):
        return None
    if backend.n_heads != 1:
        raise ValueError(f"curriculum requires one head, got {backend.head_sizes}")
    return SpeedCurriculum(
        speed_grid=backend.vars[0].grid,
        window_size=int(config.get("curriculum_window_size", 20)),
        success_threshold=float(
            config.get("curriculum_success_threshold", 0.7)
        ),
    )


def action_limits_for_backend(curriculum, backend) -> tuple[int, ...]:
    if curriculum is not None:
        return curriculum.action_limits
    return tuple(int(size) - 1 for size in backend.head_sizes)


def curriculum_metrics(curriculum) -> dict:
    if curriculum is None:
        return {}
    total_actions = max(sum(curriculum.action_counts), 1)
    metrics = {
        "curriculum/max_unlocked_idx": curriculum.max_unlocked_idx,
        "curriculum/max_unlocked_speed": curriculum.max_unlocked_speed,
        "curriculum/window_success_rate": curriculum.window_success_rate,
        "curriculum/window_size": len(curriculum.recent_results),
        "curriculum/unlock_count": len(curriculum.unlock_events),
    }
    metrics.update({
        f"curriculum/action_rate_{index}_{speed:g}": count / total_actions
        for index, (speed, count) in enumerate(
            zip(curriculum.speed_grid, curriculum.action_counts)
        )
    })
    return metrics
