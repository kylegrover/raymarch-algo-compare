"""Append-only JSONL dataset with resume support.

One row per run. On open, existing ``config_hash`` values are indexed so a
relaunched sweep skips combos already present (crash-/interrupt-resilient).
"""
from __future__ import annotations
import os
import json
from typing import Dict, Iterator, List, Optional, Set


class JsonlDataset:
    def __init__(self, path: str):
        self.path = path
        self._done: Set[str] = set()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                        ch = row.get("config_hash")
                        if ch:
                            self._done.add(ch)
                    except json.JSONDecodeError:
                        continue
        else:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)

    def has(self, config_hash: str) -> bool:
        return config_hash in self._done

    def __len__(self) -> int:
        return len(self._done)

    def append(self, row: Dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
        ch = row.get("config_hash")
        if ch:
            self._done.add(ch)

    def extend(self, rows: List[Dict]) -> None:
        """Append a batch of rows in a single open/flush (grid-scale I/O, §H).

        Crash granularity is the batch size, so callers should pick a flush
        interval that balances throughput against how much work they're willing
        to redo on an interrupt.
        """
        if not rows:
            return
        with open(self.path, "a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
        for row in rows:
            ch = row.get("config_hash")
            if ch:
                self._done.add(ch)

    @staticmethod
    def load(path: str) -> List[Dict]:
        rows: List[Dict] = []
        if not os.path.exists(path):
            return rows
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows
