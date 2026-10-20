# utils/provenance.py
"""
Proveniência de cada execução: o que rodou, com que código, em que máquina,
com que dados e com que semente. Gravado em `provenance.json` ao lado de
`metrics.json`, para que qualquer número de um artigo possa ser rastreado até
a execução que o produziu sem depender de memória nem de log de terminal.

Blocos:
  • code        — branch, commit e estado (limpo/sujo) do código. No servidor
                  não há `.git` (o rsync o exclui), então `nahas-run` grava um
                  `VERSION.json` em `src/` antes de sincronizar; aqui ele é lido.
  • environment — host, SO, Python, versões de pacotes, GPUs e driver.
  • data        — fonte, nº de linhas, primeiro e último timestamp, hash do alvo.
  • split       — tamanho e intervalo de cada partição e o embargo.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def code_version(src_dir: str = _SRC_DIR) -> Dict[str, Any]:
    """VERSION.json (escrito pelo nahas-run) ou, se houver `.git`, o próprio git."""
    vpath = os.path.join(src_dir, "VERSION.json")
    if os.path.exists(vpath):
        try:
            with open(vpath, "r", encoding="utf-8") as f:
                info = json.load(f)
            info["source"] = "VERSION.json"
            return info
        except Exception as e:  # arquivo corrompido nao pode derrubar o treino
            return {"source": "VERSION.json", "error": str(e)}
    repo = os.path.dirname(src_dir)
    try:
        def git(*args):
            return subprocess.check_output(["git", "-C", repo, *args], stderr=subprocess.DEVNULL,
                                           timeout=10).decode().strip()
        return {
            "source": "git",
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "commit_date": git("log", "-1", "--format=%cI"),
        }
    except Exception:
        return {"source": "unknown"}


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _gpus() -> list:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, timeout=10).decode().strip()
        return [dict(zip(("index", "name", "driver", "memory_total"), [p.strip() for p in line.split(",")]))
                for line in out.splitlines() if line.strip()]
    except Exception:
        return []


def environment() -> Dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {p: _pkg_version(p) for p in
                     ("tensorflow", "keras", "torch", "numpy", "pandas", "scikit-learn", "peewee")},
        "gpus": _gpus(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def data_fingerprint(df, target_col: str, timestamp_col: str = "timestamp", **meta) -> Dict[str, Any]:
    """Nº de linhas, intervalo de datas, colunas e hash do alvo (6 casas decimais)."""
    vals = np.asarray(df[target_col].values, dtype=float)
    h = hashlib.sha256(np.round(vals, 6).tobytes()).hexdigest()
    ts = df[timestamp_col] if timestamp_col in df.columns else None
    return {
        **meta,
        "n_rows": int(len(df)),
        "columns": [str(c) for c in df.columns],
        "target_column": target_col,
        "first_timestamp": str(ts.iloc[0]) if ts is not None and len(ts) else None,
        "last_timestamp": str(ts.iloc[-1]) if ts is not None and len(ts) else None,
        "target_sha256": h,
    }


def split_summary(sl: Dict[str, slice], ts_all, embargo: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"embargo": int(embargo)}
    ts = np.asarray(ts_all)
    for nome in ("train", "val", "test"):
        s = sl.get(nome)
        if s is None:
            continue
        seg = ts[s]
        out[nome] = {"n": int(len(seg)),
                     "first_timestamp": str(seg[0]) if len(seg) else None,
                     "last_timestamp": str(seg[-1]) if len(seg) else None}
    return out


def _jsonable(o):
    """Tipos numpy viram tipos nativos; o resto cai em str (datas, slices, objetos)."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def write_provenance(path: str, **blocks) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"written_at": datetime.now().isoformat(timespec="seconds"), **blocks}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=_jsonable)
    return path
