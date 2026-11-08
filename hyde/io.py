"""Crash-safe writes: write to a temp file in the same folder, then rename.

A run killed mid-write therefore never leaves a half-written result/cache/model behind
(the resume logic treats an existing file as 'done', so partial files would be fatal).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp{os.getpid()}")


def write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    tmp.write_text(text)
    os.replace(tmp, path)


def save_npz(path, compressed: bool = False, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    with open(tmp, "wb") as f:                      # file handle -> numpy does not append '.npz'
        (np.savez_compressed if compressed else np.savez)(f, **arrays)
    os.replace(tmp, path)
