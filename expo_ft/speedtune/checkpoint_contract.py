"""Serializable compatibility contract for backend-specific SpeedTune checkpoints."""

import json
import os
from pathlib import Path

from expo_ft.speedtune.runtime_config import backend_k_skip, backend_support


METADATA_FILE = "speedtune_metadata.json"
WHOLE_CHUNK_ACC_RULE = "4*vel_limit^2"


def build_checkpoint_metadata(config, backend) -> dict:
    v_min, v_max = backend_support(config, backend.name)
    curriculum_enabled = bool(config.get("curriculum_enabled", True)) and (
        backend.name in ("fixed_time", "chunk_toppra")
    )
    return {
        "version": 2,
        "exec_backend": backend.name,
        "head_sizes": [int(value) for value in backend.head_sizes],
        "n_atoms": int(config.n_atoms),
        "support": [float(v_min), float(v_max)],
        "k_skip": backend_k_skip(config, backend.name),
        "speed_grid": [[float(value) for value in var.grid] for var in backend.vars],
        "acc_rule": WHOLE_CHUNK_ACC_RULE if backend.name == "chunk_toppra" else None,
        "curriculum_config": {
            "enabled": curriculum_enabled,
            "window_size": int(config.get("curriculum_window_size", 20)),
            "success_threshold": float(
                config.get("curriculum_success_threshold", 0.7)
            ),
        },
    }


def curriculum_action_limits(metadata: dict, head_sizes) -> tuple[int, ...]:
    """Return the final training action mask stored beside a checkpoint."""
    head_sizes = tuple(int(size) for size in head_sizes)
    curriculum_config = metadata.get("curriculum_config", {})
    if not curriculum_config.get("enabled", False):
        return tuple(size - 1 for size in head_sizes)
    state = metadata.get("curriculum_state")
    if state is None:
        raise ValueError("checkpoint is missing enabled curriculum_state")
    if len(head_sizes) != 1:
        raise ValueError("speed curriculum checkpoint must have exactly one head")
    limit = int(state["max_unlocked_idx"])
    if not 0 <= limit < head_sizes[0]:
        raise ValueError(
            f"curriculum action limit {limit} outside checkpoint head {head_sizes[0]}"
        )
    return (limit,)


def write_checkpoint_metadata(checkpoint_dir, metadata: dict) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / METADATA_FILE
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def validate_checkpoint_metadata(checkpoint_dir, expected: dict) -> dict:
    path = Path(checkpoint_dir) / METADATA_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"SpeedTune checkpoint missing {METADATA_FILE}: {path}; old checkpoints are incompatible"
        )
    actual = json.loads(path.read_text(encoding="utf-8"))
    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in expected
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        raise ValueError(f"SpeedTune checkpoint contract mismatch: {mismatches}")
    return actual
