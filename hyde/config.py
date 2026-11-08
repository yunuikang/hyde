"""Config loading: default.yaml <- tuned/<ds>.yaml <- --set key=value overrides."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        if key not in node or not isinstance(node[key], dict):
            raise KeyError(f"unknown config section {dotted!r}")
        node = node[key]
    if keys[-1] not in node:
        raise KeyError(f"unknown config key {dotted!r}")
    node[keys[-1]] = value


def parse_sets(pairs: list[str]) -> list[tuple[str, object]]:
    out = []
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        out.append((key.strip(), yaml.safe_load(raw)))   # yaml gives ints/floats/bools/null
    return out


def build_config(dataset: str | None, sets: list[str] | None = None,
                 extra: list[str] | None = None, use_tuned: bool = True) -> dict:
    cfg = load_yaml(CONFIGS / "default.yaml")
    if dataset and use_tuned:
        tuned = CONFIGS / "tuned" / f"{dataset}.yaml"
        if tuned.exists():
            cfg = deep_merge(cfg, load_yaml(tuned))
    for path in extra or []:
        cfg = deep_merge(cfg, load_yaml(Path(path)))
    for key, value in parse_sets(sets or []):
        set_dotted(cfg, key, value)
    return cfg


def datasets_registry() -> dict:
    return load_yaml(CONFIGS / "datasets.yaml")
