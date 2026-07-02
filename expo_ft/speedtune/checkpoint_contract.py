"""Serializable compatibility contract for backend-specific SpeedTune checkpoints."""

import json
import os
from pathlib import Path

from expo_ft.speedtune.runtime_config import backend_k_skip, backend_support


METADATA_FILE = "speedtune_metadata.json"
WHOLE_CHUNK_ACC_RULE = "4*vel_limit^2"


def build_checkpoint_metadata(config, backend) -> dict:
    v_min, v_max = backend_support(config, backend.name)
    return {
        "version": 1,
        "exec_backend": backend.name,
        "head_sizes": [int(value) for value in backend.head_sizes],
        "n_atoms": int(config.n_atoms),
        "support": [float(v_min), float(v_max)],
        "k_skip": backend_k_skip(config, backend.name),
        "speed_grid": [[float(value) for value in var.grid] for var in backend.vars],
        "acc_rule": WHOLE_CHUNK_ACC_RULE if backend.name == "chunk_toppra" else None,
    }


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
