#!/usr/bin/env python
"""
Resume uma grade dos artigos (results/grid/<grid_id>/) numa tabela por execucao:
simbolo, modelo, normalizacao, horizonte, RMSE/MAE de teste, razao contra a
persistencia, DM, acuracia direcional, razao de aposta (mimetismo), epocas,
duracao, semente e commit. Le metrics.json + provenance.json de cada pasta.

    python analysis/resumo_grade.py results/grid/artigo1_principal_H1 [--csv saida.csv]
"""
import argparse
import glob
import json
import os
import sys

import pandas as pd


def rotulo_norm(n: dict) -> str:
    s = n.get("strategy")
    if s == "global":
        return f"global-{n.get('scaler_type', '?')}"
    if s == "local":
        return f"local-{n.get('x_mode', '?')}/{n.get('y_mode', '?')}"
    if s == "evomsn":
        return f"evomsn-k{n.get('evomsn_k_scales')}-{n.get('evomsn_predictor')}-{n.get('evomsn_short_horizon_policy', 'legacy')}"
    return str(n)


def ler_grade(grid_root: str) -> pd.DataFrame:
    linhas = []
    for mp in sorted(glob.glob(os.path.join(grid_root, "*", "metrics.json"))):
        d = os.path.dirname(mp)
        m = json.load(open(mp))
        pp = os.path.join(d, "provenance.json")
        p = json.load(open(pp)) if os.path.exists(pp) else {}
        cfg = p.get("config", {})
        b = m.get("baselines") or {}
        mod, per, mim = b.get("model", {}), b.get("persistence", {}), b.get("mimicry", {})
        dm = b.get("dm_model_vs_persistence", {})
        deg = m.get("degeneracy") or {}
        linhas.append({
            "run_id": os.path.basename(d),
            "symbol": cfg.get("symbol"), "model": p.get("model_type"),
            "norm": rotulo_norm(cfg.get("normalization", {})),
            "H": cfg.get("steps_ahead"), "L": cfg.get("window_size"),
            "rmse_test": m.get("test", {}).get("rmse"), "mae_test": m.get("test", {}).get("mae"),
            "rmse_val": m.get("validation", {}).get("rmse"),
            "rmse_persist": per.get("rmse"),
            "ratio_persist": (mod.get("rmse") / per.get("rmse")) if mod.get("rmse") and per.get("rmse") else None,
            "dm_stat": dm.get("dm_stat"), "dm_p": dm.get("p_value"),
            "dir_acc": mim.get("directional_accuracy"), "bet_ratio": mim.get("bet_size_ratio"),
            "corr_ret": mim.get("corr_returns"),
            "degen_ratio": deg.get("contribution_ratio_ensemble"),
            "epochs": p.get("epochs_trained"), "min": round((p.get("duration_s") or 0) / 60, 1),
            "seed": p.get("seed"), "commit": (p.get("code", {}).get("commit") or "")[:8],
        })
    return pd.DataFrame(linhas)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_root")
    ap.add_argument("--csv")
    a = ap.parse_args()
    df = ler_grade(a.grid_root)
    if df.empty:
        print("nenhuma execucao com metrics.json", file=sys.stderr)
        return 1
    df = df.sort_values(["symbol", "model", "norm"])
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 30)
    cols = ["symbol", "model", "norm", "rmse_test", "rmse_persist", "ratio_persist", "dm_stat", "dir_acc", "bet_ratio", "epochs", "min"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\n== razao RMSE/persistencia (teste): linhas = normalizacao, colunas = simbolo x modelo ==")
    piv = df.pivot_table(index="norm", columns=["symbol", "model"], values="ratio_persist")
    print(piv.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\n== melhor normalizacao por simbolo x modelo (teste) ==")
    best = df.loc[df.groupby(["symbol", "model"])["rmse_test"].idxmin(), ["symbol", "model", "norm", "ratio_persist"]]
    print(best.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if a.csv:
        df.to_csv(a.csv, index=False)
        print(f"\ncsv: {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
