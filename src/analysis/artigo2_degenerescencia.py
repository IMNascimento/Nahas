#!/usr/bin/env python
"""
Artigo 2 — degenerescencia de horizonte curto na EvoMSN: agrega os metrics.json
e provenance.json da grade `artigo2_degenerescencia` (legacy vs window_stats,
K in {2,3,4}, preditor linear/mlp, H in {1,6,24}, BTC e ETH, LSTM e Transformer)
e calcula o teste de invariancia de backbone D (LSTM vs Transformer sob a mesma
normalizacao) a partir dos CSVs de teste exportados. Nenhum treino novo.

Saidas em --out:
  resultados.csv  uma linha por execucao: periodos, escalas degeneradas, std do
                  alvo normalizado por escala, r (ensemble e por escala), RMSE,
                  MAE, persistencia, DM, mimetismo, epocas, semente, commit.
  D.csv           uma linha por (symbol, H, K, predictor, policy): D e n.

    python analysis/artigo2_degenerescencia.py --out results/artigo2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.degeneracy import backbone_invariance_from_csv   # noqa: E402

GRID = "artigo2_degenerescencia"
CHAVES = ["symbol", "H", "K", "predictor", "policy"]


def carregar(grid_root: str) -> pd.DataFrame:
    linhas = []
    for p in sorted(glob.glob(os.path.join(grid_root, "*", "provenance.json"))):
        d = os.path.dirname(p)
        if not os.path.exists(os.path.join(d, "metrics.json")):
            continue
        prov = json.load(open(p))
        m = json.load(open(os.path.join(d, "metrics.json")))
        cfg, norm = prov["config"], prov["config"]["normalization"]
        if norm.get("strategy") != "evomsn":
            continue
        deg, fr = m.get("degeneracy", {}), m.get("degeneracy", {}).get("fit_report", {})
        base = m.get("baselines", {})
        dm = base.get("dm_model_vs_persistence", {})
        mim = base.get("mimicry", {})
        te = [c for c in glob.glob(os.path.join(d, "csv", "results_*.csv")) if "results_val_" not in c]
        periods = list(fr.get("periods", deg.get("periods", [])))
        degen = list(fr.get("degenerate_scales", []))
        std = list(fr.get("backbone_target_std_per_scale", []))
        r_esc = list(deg.get("contribution_ratio_per_scale", []))
        linhas.append({
            "run_dir": os.path.basename(d), "symbol": cfg["symbol"], "model": prov["model_type"],
            "H": int(cfg["steps_ahead"]), "K": int(norm["evomsn_k_scales"]),
            "predictor": norm["evomsn_predictor"], "policy": norm["evomsn_short_horizon_policy"],
            "periods": "/".join(str(int(x)) for x in periods),
            "n_degenerate": int(sum(bool(x) for x in degen)),
            "n_affected": int(sum(int(x) > int(cfg["steps_ahead"]) for x in periods)),
            "target_std": "/".join(f"{x:.3g}" for x in std),
            "target_std_min": float(min(std)) if std else np.nan,
            "target_std_max": float(max(std)) if std else np.nan,
            "r_ens": deg.get("contribution_ratio_ensemble", np.nan),
            "r_scales": "/".join(f"{x:.3g}" for x in r_esc),
            "r_min": float(np.nanmin(r_esc)) if r_esc else np.nan,
            "r_max": float(np.nanmax(r_esc)) if r_esc else np.nan,
            "eps": deg.get("eps", np.nan),
            "rmse_test": m["test"]["rmse"], "mae_test": m["test"]["mae"],
            "rmse_persist": base.get("persistence", {}).get("rmse", np.nan),
            "mae_persist": base.get("persistence", {}).get("mae", np.nan),
            "ratio_persist": m["test"]["rmse"] / base.get("persistence", {}).get("rmse", np.nan),
            "dm_stat": dm.get("dm_stat", np.nan), "dm_p": dm.get("p_value", np.nan),
            "dir_acc": mim.get("directional_accuracy", np.nan), "bet_ratio": mim.get("bet_size_ratio", np.nan),
            "corr_returns": mim.get("corr_returns", np.nan),
            "seed": cfg.get("seed"), "commit": (prov.get("code") or {}).get("commit", "")[:7],
            "test_csv": te[0] if te else "",
        })
    return pd.DataFrame(linhas)


def invariancia(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for chave, g in df.groupby(CHAVES):
        a = g[g["model"] == "lstm"]
        b = g[g["model"] == "transformer"]
        if len(a) != 1 or len(b) != 1 or not a.iloc[0]["test_csv"] or not b.iloc[0]["test_csv"]:
            continue
        r = backbone_invariance_from_csv(a.iloc[0]["test_csv"], b.iloc[0]["test_csv"])
        out.append({**dict(zip(CHAVES, chave)), "D": r["D"], "n": r["n"],
                    "periods": a.iloc[0]["periods"], "n_affected": a.iloc[0]["n_affected"]})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "grid", GRID))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = carregar(a.grid_root)
    df.to_csv(os.path.join(a.out, "resultados.csv"), index=False)
    dD = invariancia(df)
    dD.to_csv(os.path.join(a.out, "D.csv"), index=False)
    print(f"{len(df)} execucoes, {len(dD)} pares LSTM/Transformer -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
