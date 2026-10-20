"""
Testes da proveniência por execução.

Rodar de dentro de `src/`:
    ./venv/bin/python -m pytest tests/test_provenance.py -q
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.provenance import (                                     # noqa: E402
    code_version,
    data_fingerprint,
    environment,
    split_summary,
    write_provenance,
)


def test_code_version_le_o_version_json_do_servidor(tmp_path):
    (tmp_path / "VERSION.json").write_text(json.dumps({"branch": "run/servidor", "commit": "abc123", "dirty": False}))
    v = code_version(str(tmp_path))
    assert v["source"] == "VERSION.json" and v["commit"] == "abc123"


def test_code_version_sem_version_json_cai_no_git_ou_unknown(tmp_path):
    v = code_version(str(tmp_path / "src_inexistente"))
    assert v["source"] in ("git", "unknown")
    v2 = code_version()                       # este repositório: tem .git
    assert v2["source"] in ("git", "VERSION.json") and ("commit" in v2 or "error" in v2)


def test_environment_tem_os_blocos_minimos():
    e = environment()
    for k in ("hostname", "platform", "python", "packages", "gpus", "cuda_visible_devices"):
        assert k in e
    assert e["packages"]["numpy"] is not None


def test_fingerprint_dos_dados_e_deterministico_e_sensivel():
    ts = pd.date_range("2024-01-01", periods=10, freq="h")
    df = pd.DataFrame({"timestamp": ts, "close": np.arange(10.0), "volume": 1.0})
    a = data_fingerprint(df, "close", symbol="X")
    b = data_fingerprint(df.copy(), "close", symbol="X")
    assert a == b
    assert a["n_rows"] == 10 and a["symbol"] == "X"
    assert a["first_timestamp"].startswith("2024-01-01 00:00") and a["last_timestamp"].startswith("2024-01-01 09:00")
    df2 = df.copy()
    df2.loc[3, "close"] += 1e-3
    assert data_fingerprint(df2, "close")["target_sha256"] != a["target_sha256"]


def test_resumo_da_particao():
    ts = np.array(pd.date_range("2024-01-01", periods=100, freq="h").astype(str))
    sl = {"train": slice(0, 60), "val": slice(65, 80), "test": slice(85, 100)}
    s = split_summary(sl, ts, embargo=5)
    assert s["embargo"] == 5 and s["train"]["n"] == 60 and s["test"]["n"] == 15
    assert s["val"]["first_timestamp"] == ts[65]


def test_write_provenance_serializa_tipos_numpy(tmp_path):
    p = write_provenance(str(tmp_path / "prov.json"), seed=np.int64(7), arr=np.array([1.0]), when=pd.Timestamp("2024-01-01"))
    d = json.load(open(p))
    assert d["seed"] == 7 and "written_at" in d and "when" in d
