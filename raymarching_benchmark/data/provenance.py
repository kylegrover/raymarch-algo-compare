"""Run provenance + stable config hashing for the sweep dataset.

The config hash identifies a run (scene/strategy/budget/resolution/threshold) so
the sweep can skip already-completed combos on restart. Provenance (git SHA,
GPU, host, timestamp) is recorded on every row so the dataset is self-describing
months later.
"""
from __future__ import annotations
import os
import sys
import json
import hashlib
import socket
import datetime
import subprocess
from typing import Dict, Optional


def _git(args) -> Optional[str]:
    try:
        out = subprocess.run(["git"] + args, capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def git_info() -> Dict[str, object]:
    sha = _git(["rev-parse", "HEAD"])
    status = _git(["status", "--porcelain"])
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def config_hash(config: Dict) -> str:
    """Stable short hash of a run config (order-independent)."""
    blob = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def provenance(gpu: Optional[Dict] = None) -> Dict[str, object]:
    return {
        "git": git_info(),
        "gpu": gpu or {},
        "host": socket.gethostname(),
        "platform": sys.platform,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
    }
