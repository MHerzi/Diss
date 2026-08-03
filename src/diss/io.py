"""Deterministic hashing and reproducible experiment artifacts."""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from .errors import DissError


def _normal_form(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.tolist(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return _normal_form(asdict(value))
    if isinstance(value, dict):
        return {str(key): _normal_form(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_normal_form(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def hash_value(value: Any) -> str:
    encoded = json.dumps(
        _normal_form(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def current_git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def current_git_dirty() -> bool | None:
    try:
        output = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        return bool(output.strip())
    except (OSError, subprocess.SubprocessError):
        return None


def create_manifest(plan: Any) -> dict[str, Any]:
    config_hash = hash_value(plan.config)
    data_hash = hash_value(
        {
            "returnsHash": hash_array(plan.data.returns),
            "observationIndex": plan.data.observation_index,
            "time": plan.data.time,
            "variableNames": plan.data.variable_names,
        }
    )
    experiment_hash = hash_value({"configHash": config_hash, "dataHash": data_hash})
    return {
        "schemaVersion": plan.config["schemaVersion"],
        "experimentId": experiment_hash[:16],
        "gitCommit": current_git_commit(),
        "gitDirty": current_git_dirty(),
        "pythonVersion": plan.metadata["pythonVersion"],
        "created": plan.metadata["created"],
        "randomSeed": plan.metadata["randomSeed"],
        "configHash": config_hash,
        "dataHash": data_hash,
        "observationCount": plan.data.observation_count,
        "seriesCount": plan.data.series_count,
        "products": {"numpy": np.__version__, "scipy": scipy.__version__},
    }


def save_run(results: Any, output_root: str | Path = "results") -> Path:
    """Save a trusted local pickle plus a human-readable JSON manifest."""

    manifest = getattr(results, "manifest", None)
    if not manifest or "experimentId" not in manifest:
        raise DissError(
            "diss:io:MissingManifest", "Only canonical experiment results can be saved."
        )
    root = Path(output_root)
    run_name = f"{datetime.now(UTC):%Y%m%dT%H%M%S_%f}_{manifest['experimentId']}"
    directory = root / run_name
    directory.mkdir(parents=True, exist_ok=False)
    with (directory / "results.pkl").open("wb") as stream:
        pickle.dump(results, stream, protocol=pickle.HIGHEST_PROTOCOL)
    (directory / "manifest.json").write_text(
        json.dumps(_normal_form(manifest), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return directory


def load_run(run_directory: str | Path) -> Any:
    """Load a result produced locally by :func:`save_run`.

    Pickle files must only be loaded from trusted experiment directories.
    """

    with (Path(run_directory) / "results.pkl").open("rb") as stream:
        return pickle.load(stream)
