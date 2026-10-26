#!/usr/bin/env python
"""
Artigo 6 — tabelas (.tex) a partir da extracao de literatura
(<artigo>/extraction/papers.csv + rw_reference.csv) e, quando a grade
artigo6_pipelines existir, da reavaliacao dos pipelines.

    python analysis/artigo6_tabelas.py --art <pasta do artigo no Overleaf> [--grid-root results/grid]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

# chaves do .bib comum que ja existiam com outro nome
KEYMAP = {"Seabe2023BiLSTM": "fractalfract7020203"}
ASSET = {"BTC": "BTC", "ETH": "ETH", "LTC": "LTC", "XRP": "XRP"}


def esc(s):
    return str(s).replace("%", "\\%").replace("&", "\\&").replace("_", "\\_")


def tab_papers(art):
    p = pd.read_csv(os.path.join(art, "extraction", "papers.csv"))
    r = pd.read_csv(os.path.join(art, "extraction", "rw_reference.csv"))
    rows = []
    n_assess = n_better = n_worse = n_ind = 0
    for _, a in p.iterrows():
        key = KEYMAP.get(a.key, a.key)
        rr = r[r.key == a.key]
        freq = {"daily": "d", "hourly": "h"}.get(str(a.frequency).split()[0], str(a.frequency)[:1])
        if rr.empty or rr.verdict.str.startswith("not assessable").all():
            # uma linha por artigo
            reason = rr.verdict.iloc[0] if not rr.empty else str(a.rw_verdict)
            reason = reason.replace("not assessable", "n/a").replace("normalized units", "normalized").replace("asset not in local DB; ", "")
            metric = "normalized scale" if "normalized" in reason else "see sheet"
            period = "declared" if str(a.test_period_declared).startswith("yes") else ("partial" if str(a.test_period_declared).startswith("partial") else "NR")
            rows.append(f"\\citet{{{key}}} & {esc(a.assets).split('(')[0].strip()[:18]} & {freq} & {period} & {metric} & -- & {esc(reason)} \\\\")
            continue
        for _, x in rr.iterrows():
            n_assess += 1
            v = x.verdict
            n_better += v == "better"; n_worse += v == "worse"; n_ind += v == "indistinguishable"
            pm = esc(x.paper_metrics).replace("=", " ")
            # mesma metrica da persistencia: RMSE quando o artigo da RMSE, senao MAPE/MAE
            if "RMSE" in str(x.paper_metrics):
                rw = f"RMSE {x.rw_RMSE:,.1f}"
            elif "MAPE" in str(x.paper_metrics):
                rw = f"MAPE {x.rw_MAPE_pct:.2f}\\%"
            else:
                rw = f"MAE {x.rw_MAE:,.1f}"
            vlab = {"better": "better", "worse": "worse", "indistinguishable": "within 5\\%"}[v]
            rows.append(f"\\citet{{{key}}} & {x.asset} & {freq} & {x.test_start} to {x.test_end} & {pm} & {rw} & {vlab} \\\\")
    linhas = ["\\begin{table}[p]", "\\centering", "\\scriptsize",
              "\\caption{Published pipelines and the naive reference on the same period. One row per assessable paper--asset pair and one row per paper whose test period is not declared or whose metrics are on a normalized scale. "
              "Fr.: d = daily, h = hourly; horizon one step in every row. The random-walk value is computed on the declared test period from the Binance USDT series (daily close = last hourly close of the UTC day). "
              "A reported value within $\\pm$5\\% of the persistence value is marked \\emph{within 5\\%} (indistinguishable). Full extraction in \\texttt{extraction/papers.csv}.}",
              "\\label{tab:papers}", "\\begin{tabular}{@{}p{2.6cm}p{1.5cm}p{0.45cm}p{2.5cm}>{\\raggedright\\arraybackslash}p{2.9cm}p{1.5cm}>{\\raggedright\\arraybackslash}p{1.9cm}@{}}", "\\toprule",
              "Paper & Asset & Fr. & Test period & Reported & Random walk & Verdict \\\\", "\\midrule", *rows, "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    os.makedirs(os.path.join(art, "tables"), exist_ok=True)
    open(os.path.join(art, "tables", "tab_papers.tex"), "w").write("\n".join(linhas) + "\n")
    print(f"papers: {len(p)} artigos, {n_assess} pares avaliaveis: better {n_better}, worse {n_worse}, indistinguishable {n_ind}")
    return p, r


def tab_pipelines(art, grid_root):
    """Reavaliacao dos pipelines (quando a grade existir)."""
    rows = []
    for mp in glob.glob(os.path.join(grid_root, "artigo6_pipelines", "*", "metrics.json")):
        d = os.path.dirname(mp)
        if not os.path.exists(os.path.join(d, "provenance.json")):
            continue
        m, pv = json.load(open(mp)), json.load(open(os.path.join(d, "provenance.json")))
        b, cfg = m["baselines"], pv["config"]
        cz = b.get("constant_zero_normalized", {})
        rows.append({"symbol": cfg["symbol"], "model": pv["model_type"], "norm": cfg["normalization"].get("y_mode") or cfg["normalization"].get("scaler_type"),
                     "seed": pv["seed"], "rmse": b["model"]["rmse"], "rmse_persist": b["persistence"]["rmse"], "U": b["model"]["rmse"] / b["persistence"]["rmse"],
                     "corr": b["mimicry"]["corr_returns"], "bet": b["mimicry"]["bet_size_ratio"], "da": b["mimicry"]["directional_accuracy"],
                     "dm": b["dm_model_vs_persistence"]["dm_stat"], "rmse_zero": cz.get("rmse"), "zero_eq_persist": cz.get("equals_persistence")})
    df = pd.DataFrame(rows)
    if df.empty:
        print("pipelines: grade artigo6_pipelines ausente")
        return df
    df.to_csv(os.path.join(art, "extraction", "pipelines_runs.csv"), index=False)
    print(f"pipelines: {len(df)} execucoes")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--grid-root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "grid"))
    a = ap.parse_args()
    tab_papers(a.art)
    tab_pipelines(a.art, a.grid_root)


if __name__ == "__main__":
    main()
