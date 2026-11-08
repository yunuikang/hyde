"""Dataset loading: CSV -> cleaned numeric/categorical arrays (+ cache) and stratified splits.

One rule set for all 11 datasets (see configs/datasets.yaml):
  * drop label / metadata / leaky columns (IPs, ports, timestamps, NSL-KDD difficulty)
  * a column is numeric iff every non-empty train value parses as a float (full scan)
  * protocol/service/flag (+ per-dataset list) are categorical; >4096 levels -> dropped
  * labels: strip, benign aliases -> "benign", cicids2017 web-attack spelling unified,
    test-only labels -> -1 ("unknown")
The parsed result is cached as cache/<ds>.npz so the CSV is read once.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ROOT, datasets_registry
from .io import save_npz

CACHE = ROOT / "cache"
NA_TOKEN = "<NA>"
INF_TOKENS = ["inf", "+inf", "-inf", "infinity", "+infinity", "-infinity", "nan"]
UNKNOWN = -1


@dataclass
class Dataset:
    name: str
    X_num: np.ndarray            # (N, Fn) float32, NaN = missing   (train file)
    X_cat: np.ndarray            # (N, Fc) int32, code == len(vocab) -> unseen/unknown
    y: np.ndarray                # (N,) int64 class index (benign = 0)
    X_num_test: np.ndarray
    X_cat_test: np.ndarray
    y_test: np.ndarray           # -1 = label never seen in train
    classes: list[str]           # index -> name
    num_names: list[str]
    cat_names: list[str]
    vocabs: list[list[str]]      # per categorical column
    meta: dict = field(default_factory=dict)

    @property
    def cat_widths(self) -> list[int]:
        return [len(v) + 1 for v in self.vocabs]   # +1 unknown slot


# --------------------------------------------------------------------------- names / labels
def normalize_name(name: str) -> str:
    return "_".join(name.strip().lower().replace("-", "_").split())


def canon_cicids2017(label: str) -> str:
    """Train says 'brute force', test says 'web attack � brute force' -> one name."""
    cleaned = label.lower()
    for token in ("�", "–", "—", "-", ":", "/"):
        cleaned = cleaned.replace(token, " ")
    cleaned = normalize_name(cleaned)
    aliases = {
        "brute_force": "web attack brute force",
        "web_attack_brute_force": "web attack brute force",
        "xss": "web attack xss",
        "web_attack_xss": "web attack xss",
        "sql_injection": "web attack sql injection",
        "web_attack_sql_injection": "web attack sql injection",
    }
    return aliases.get(cleaned, label)


def clean_labels(raw, benign_aliases: set[str], canon: str | None) -> np.ndarray:
    out = []
    for value in raw:
        label = ("" if value is None else str(value)).strip()
        if label.lower() in benign_aliases:
            label = "benign"
        elif canon == "cicids2017":
            label = canon_cicids2017(label)
        out.append(label)
    return np.asarray(out, dtype=object)


# --------------------------------------------------------------------------- CSV parsing
def _unique_header(path: Path) -> list[str]:
    with open(path, newline="") as f:
        header = next(csv.reader(f))
    seen: dict[str, int] = {}
    out = []
    for name in header:
        if name in seen:                      # e.g. cicids2017 'fwd header length' twice
            seen[name] += 1
            out.append(f"{name}__dup{seen[name]}")
        else:
            seen[name] = 0
            out.append(name)
    return out


def _read(path: Path, threads: int, as_text: set[str] = frozenset()):
    """Read a CSV; columns whose normalised name is in `as_text` stay raw strings
    (categorical codes must match the literal CSV text that infer.py will see)."""
    import os
    os.environ.setdefault("POLARS_MAX_THREADS", str(threads))
    import polars as pl
    header = _unique_header(path)
    overrides = {h: pl.Utf8 for h in header if normalize_name(h) in as_text}
    return pl.read_csv(path, has_header=True, new_columns=header, schema_overrides=overrides,
                       infer_schema_length=None, low_memory=False)


def _as_float(series):
    """Parse a column to float64. Returns (values, n_unparseable)."""
    import polars as pl
    if series.dtype.is_numeric() or series.dtype == pl.Boolean:
        return series.cast(pl.Float64), 0
    s = series.cast(pl.Utf8).str.strip_chars()
    lower = s.str.to_lowercase()
    is_inf_token = lower.is_in(INF_TOKENS)
    f = s.cast(pl.Float64, strict=False)
    bad = (s.is_not_null() & (s != "") & f.is_null() & ~is_inf_token).sum()
    f = pl.when(is_inf_token).then(None).otherwise(f)
    return pl.select(f.alias("v")).to_series(), int(bad)


def _as_category(series):
    import polars as pl
    return series.cast(pl.Utf8).str.strip_chars().fill_null(NA_TOKEN)


def _stratified_cap(labels: np.ndarray, cap: int, seed: int) -> np.ndarray:
    keep = stratified_subset(np.arange(labels.size), labels, max_rows=cap, seed=seed)
    return np.sort(keep)


def build_cache(name: str, threads: int = 6, verbose: bool = True) -> Path:
    import polars as pl
    reg = datasets_registry()
    spec = reg["datasets"][name]
    root = Path(reg["root"]) / spec["dir"]
    t0 = time.time()
    log = (lambda *a: print(f"[data:{name}]", *a, flush=True)) if verbose else (lambda *a: None)

    benign = {a.lower() for a in reg["benign_aliases"]}
    drop_label = {normalize_name(c) for c in reg["drop_label_columns"]}
    drop_meta = {normalize_name(c) for c in reg["drop_metadata"]}
    drop_leaky = {normalize_name(c) for c in reg["drop_leaky"]}
    suffixes = tuple(reg["drop_metadata_suffix"])
    forced_cat = {normalize_name(c) for c in reg["always_categorical"] + spec.get("categorical", [])}

    log("reading train CSV ...")
    tr = _read(root / "training-flow.csv", threads, forced_cat)
    label_col = next(c for c in tr.columns if normalize_name(c) == reg["label_column"])
    y_tr_names = clean_labels(tr[label_col].to_list(), benign, spec.get("label_canon"))
    if spec.get("train_cap"):
        keep = _stratified_cap(y_tr_names, int(spec["train_cap"]), seed=42)
        tr = tr[keep]
        y_tr_names = y_tr_names[keep]
        log(f"train capped to {keep.size:,} rows (stratified)")

    dropped: dict[str, str] = {}
    num_cols, cat_cols, vocabs = [], [], []
    num_arrays = []
    for col in tr.columns:
        n = normalize_name(col)
        if col == label_col:
            continue
        if n in drop_label:
            dropped[col] = "label"; continue
        if n in drop_meta or n.endswith(suffixes):
            dropped[col] = "metadata"; continue
        if n in drop_leaky or normalize_name(col.split("__dup")[0]) in drop_leaky:
            dropped[col] = "leaky"; continue
        series = tr[col]
        if n not in forced_cat:
            values, bad = _as_float(series)
            if bad == 0:
                if values.drop_nulls().n_unique() <= 1:
                    dropped[col] = "constant"; continue
                num_cols.append(col)
                num_arrays.append(values.to_numpy().astype(np.float32))
                continue
        cats = _as_category(series)
        vocab = sorted(cats.unique().to_list())
        if len(vocab) <= 1:
            dropped[col] = "constant"; continue
        if len(vocab) > reg["max_categories"]:
            dropped[col] = f"high_cardinality({len(vocab)})"; continue
        cat_cols.append(col)
        vocabs.append(vocab)

    X_num = np.stack(num_arrays, axis=1) if num_arrays else np.zeros((len(tr), 0), np.float32)
    X_num[~np.isfinite(X_num)] = np.nan
    del num_arrays

    def encode_cats(frame) -> np.ndarray:
        if not cat_cols:
            return np.zeros((len(frame), 0), np.int32)
        out = np.empty((len(frame), len(cat_cols)), np.int32)
        for j, (col, vocab) in enumerate(zip(cat_cols, vocabs)):
            cats = _as_category(frame[col])
            out[:, j] = cats.replace_strict(vocab, list(range(len(vocab))),
                                            default=len(vocab), return_dtype=pl.Int32).to_numpy()
        return out

    X_cat = encode_cats(tr)
    del tr

    # class index: benign first, then by train frequency (desc), ties by name
    names, counts = np.unique(y_tr_names, return_counts=True)
    order = sorted(zip(names, counts), key=lambda t: (t[0] != "benign", -t[1], t[0]))
    classes = [str(c) for c, _ in order]
    index = {c: i for i, c in enumerate(classes)}
    y = np.asarray([index[c] for c in y_tr_names], dtype=np.int64)
    log(f"train: {X_num.shape[0]:,} rows, {len(num_cols)} numeric + {len(cat_cols)} categorical, "
        f"{len(classes)} classes ({time.time() - t0:.0f}s)")

    log("reading test CSV ...")
    te = _read(root / "test-flow.csv", threads, forced_cat)
    te_label = next(c for c in te.columns if normalize_name(c) == reg["label_column"])
    y_te_names = clean_labels(te[te_label].to_list(), benign, spec.get("label_canon"))
    missing = [c for c in num_cols + cat_cols if c not in te.columns]
    if missing:
        raise ValueError(f"test file lacks columns {missing}")
    X_num_te = np.stack([_as_float(te[c])[0].to_numpy().astype(np.float32) for c in num_cols], axis=1) \
        if num_cols else np.zeros((len(te), 0), np.float32)
    X_num_te[~np.isfinite(X_num_te)] = np.nan
    X_cat_te = encode_cats(te)
    y_te = np.asarray([index.get(c, UNKNOWN) for c in y_te_names], dtype=np.int64)
    test_only = sorted({c for c in y_te_names if c not in index})
    del te

    meta = {
        "name": name, "classes": classes, "num_names": num_cols, "cat_names": cat_cols,
        "vocabs": vocabs, "dropped": dropped, "test_only_labels": test_only,
        "label_canon": spec.get("label_canon"), "train_cap": spec.get("train_cap"),
        "n_train": int(y.size), "n_test": int(y_te.size),
        "train_counts": {c: int((y == i).sum()) for i, c in enumerate(classes)},
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.npz"
    save_npz(path, X_num=X_num, X_cat=X_cat, y=y, X_num_test=X_num_te, X_cat_test=X_cat_te,
             y_test=y_te, meta=np.asarray(json.dumps(meta)))
    log(f"test: {y_te.size:,} rows, test-only labels: {test_only or 'none'}")
    log(f"dropped: {dropped}")
    log(f"cache written: {path} ({time.time() - t0:.0f}s)")
    return path


def load_dataset(name: str, threads: int = 6, rebuild: bool = False) -> Dataset:
    path = CACHE / f"{name}.npz"
    if rebuild or not path.exists():
        build_cache(name, threads)
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    return Dataset(name=name, X_num=z["X_num"], X_cat=z["X_cat"], y=z["y"],
                   X_num_test=z["X_num_test"], X_cat_test=z["X_cat_test"], y_test=z["y_test"],
                   classes=meta["classes"], num_names=meta["num_names"],
                   cat_names=meta["cat_names"], vocabs=meta["vocabs"], meta=meta)


# --------------------------------------------------------------------------- splits
def split_fit_val(y: np.ndarray, val_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Stratified split. Each class with n >= 2 sends round(frac*n) (>=1, <=n-1) rows to val;
    singleton classes stay in fit."""
    rng = np.random.default_rng(seed)
    fit, val = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        if idx.size < 2:
            fit.append(idx); continue
        n_val = min(max(1, int(round(val_frac * idx.size))), idx.size - 1)
        val.append(idx[:n_val]); fit.append(idx[n_val:])
    return np.sort(np.concatenate(fit)), np.sort(np.concatenate(val)) if val else np.zeros(0, int)


def stratified_subset(idx: np.ndarray, y: np.ndarray, max_rows: int | None = None,
                      frac: float | None = None, seed: int = 0, min_per_class: int = 50) -> np.ndarray:
    """Proportional per-class subsample of `idx`, keeping >= min(n_c, min_per_class) rows/class."""
    target = max_rows if max_rows is not None else int(round((frac or 1.0) * idx.size))
    if target >= idx.size:
        return np.sort(idx)
    rng = np.random.default_rng(seed)
    labels = y[idx]
    keep = []
    for c in np.unique(labels):
        members = idx[labels == c]
        n = max(min(members.size, min_per_class), int(round(members.size * target / idx.size)))
        keep.append(rng.choice(members, size=min(n, members.size), replace=False))
    return np.sort(np.concatenate(keep))
