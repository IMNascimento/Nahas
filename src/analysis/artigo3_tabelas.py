#!/usr/bin/env python
"""
Tabelas (LaTeX, booktabs) e figuras (PDF) do artigo 3 a partir de
results/artigo3/resultados.csv e aci_alpha_paths.npz.

    python analysis/artigo3_tabelas.py --res results/artigo3 --out <pasta do artigo no Overleaf>

Escreve <out>/tables/tab*.tex e <out>/figures/fig*.pdf, e imprime os numeros
de manchete usados no texto.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
ASSET_LABEL = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL", "BNBUSDT": "BNB"}
NORMS = ["global", "window", "anchored"]
NORM_LABEL = {"global": "Global Z-score", "window": "Window Z-score", "anchored": "Anchored"}
CONS = ["split", "lsc", "aci_raw", "aci_lsc"]
CON_LABEL = {"split": "Split (raw)", "lsc": "Locally scaled", "aci_raw": "ACI (raw)", "aci_lsc": "ACI (scaled)"}
MODELS = ["lstm", "transformer", "linear"]
MODEL_LABEL = {"lstm": "LSTM", "transformer": "Transformer", "linear": "Linear", "persistence": "Persistence"}
# Paleta categorica validada (dataviz): ordem fixa por construcao
COLORS = {"split": "#2a78d6", "lsc": "#eb6834", "aci_raw": "#1baf7a", "aci_lsc": "#eda100",
          "global": "#2a78d6", "window": "#1baf7a", "anchored": "#eb6834"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
                     "legend.frameon": False, "figure.dpi": 150})


def f3(x):
    return "--" if not np.isfinite(x) else f"{x:.3f}"


def pct(x):
    return "$\\infty$" if not np.isfinite(x) else f"{100 * x:.2f}"


def tabela(df_rows, header, caption, label, out, col_fmt, extra_note="", wide=False):
    """`wide=True` encolhe a tabela a largura do texto (tabelas com muitas colunas)."""
    abre = ["\\resizebox{\\textwidth}{!}{%"] if wide else []
    fecha = ["}"] if wide else []
    linhas = ["\\begin{table}[t]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
              *abre, f"\\begin{{tabular}}{{{col_fmt}}}", "\\toprule", header, "\\midrule", *df_rows, "\\bottomrule",
              "\\end{tabular}", *fecha]
    if extra_note:
        linhas.append(f"\\par\\smallskip\\raggedright\\footnotesize {extra_note}")
    linhas.append("\\end{table}")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")


def ultimo_passo(df):
    return df[df["step"] == df["H"]]


def tab1(df, out, model="lstm", nome="tab1_h1", caption_extra=""):
    """H=1, alpha=0.10: split vs locally scaled, por ativo e normalizacao, + persistencia."""
    d = df[(df.H == 1) & (df.alpha == 0.10) & (df.model.isin([model, "persistence"]))]
    rows = []
    for a in ASSETS:
        blocos = []
        for n in NORMS:
            for c in ("split", "lsc"):
                r = d[(d.symbol == a) & (d.norm == n) & (d.model == model) & (d.construction == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                blocos.append(f"& {NORM_LABEL[n]} & {CON_LABEL[c]} & {f3(r.coverage)} & {pct(r.rel_width)} & {pct(r.is_rel)} & {f3(r.gap)} \\\\")
        for c in ("split", "lsc"):
            r = d[(d.symbol == a) & (d.model == "persistence") & (d.construction == c)]
            if not r.empty:
                r = r.iloc[0]
                blocos.append(f"& Persistence & {CON_LABEL[c]} & {f3(r.coverage)} & {pct(r.rel_width)} & {pct(r.is_rel)} & {f3(r.gap)} \\\\")
        if blocos:
            blocos[0] = f"\\multirow{{{len(blocos)}}}{{*}}{{{ASSET_LABEL[a]}}} " + blocos[0]
            rows += blocos + ["\\midrule"]
    rows = rows[:-1]
    tabela(rows, "Asset & Normalization & Interval & Coverage & Width (\\%) & IS (\\%) & Gap \\\\",
           f"Split conformal interval with the raw residual against the locally scaled interval, {MODEL_LABEL[model]} backbone, "
           f"$H = 1$, $\\alpha = 0.10$. Width and interval score (IS) are in percent of the price at the forecast origin; "
           f"gap is the coverage in the low volatility tercile minus the coverage in the high tercile. The persistence rows use the anchored scale.{caption_extra}",
           f"tab:{nome}", out, "llllrrr",
           "Under global Z-score the two constructions coincide by construction.")


def tab2(df, out):
    """Tercis de volatilidade, H=1, LSTM, alpha=0.10, split vs lsc."""
    d = df[(df.H == 1) & (df.alpha == 0.10) & (df.model == "lstm")]
    rows = []
    for a in ASSETS:
        blocos = []
        for n in NORMS:
            for c in ("split", "lsc"):
                r = d[(d.symbol == a) & (d.norm == n) & (d.construction == c)]
                if r.empty:
                    continue
                r = r.iloc[0]
                blocos.append(f"& {NORM_LABEL[n]} & {CON_LABEL[c]} & {f3(r.cov_T1)} & {f3(r.cov_T2)} & {f3(r.cov_T3)} & {f3(r.gap)} & {pct(r.width_T1)} & {pct(r.width_T3)} \\\\")
        blocos[0] = f"\\multirow{{{len(blocos)}}}{{*}}{{{ASSET_LABEL[a]}}} " + blocos[0]
        rows += blocos + ["\\midrule"]
    rows = rows[:-1]
    tabela(rows, "Asset & Normalization & Interval & Cov.\\ low & Cov.\\ mid & Cov.\\ high & Gap & Width low (\\%) & Width high (\\%) \\\\",
           "Coverage and width within terciles of causal realized volatility (24 hourly log returns ending at the forecast origin), "
           "LSTM backbone, $H = 1$, $\\alpha = 0.10$.", "tab:terciles", out, "lllrrrrrr", wide=True)


def tab3(df, out):
    """Horizontes: ancorada, LSTM, ultimo passo, ambas as alfas, split vs lsc."""
    d = ultimo_passo(df[(df.model == "lstm") & (df.norm == "anchored")])
    rows = []
    for a in ASSETS:
        blocos = []
        for H in (1, 6, 24):
            for al in (0.10, 0.05):
                cel = []
                for c in ("split", "lsc"):
                    r = d[(d.symbol == a) & (d.H == H) & (d.alpha == al) & (d.construction == c)].iloc[0]
                    cel.append(f"{f3(r.coverage)} & {pct(r.rel_width)} & {f3(r.gap)}")
                blocos.append(f"& {H} & {al:.2f} & " + " & ".join(cel) + " \\\\")
        blocos[0] = f"\\multirow{{{len(blocos)}}}{{*}}{{{ASSET_LABEL[a]}}} " + blocos[0]
        rows += blocos + ["\\midrule"]
    rows = rows[:-1]
    tabela(rows, "Asset & $H$ & $\\alpha$ & \\multicolumn{3}{c}{Split (raw)} & \\multicolumn{3}{c}{Locally scaled} \\\\\n"
                 "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9} & & & Cov. & Width (\\%) & Gap & Cov. & Width (\\%) & Gap \\\\",
           "Anchored normalization, LSTM backbone, last step $h = H$ of each horizon, both levels.",
           "tab:horizons", out, "lllrrrrrr")


def tab4(df, out):
    """ACI, H=1, LSTM, alpha=0.10: cobertura, largura finita, horas infinitas, gap."""
    d = df[(df.H == 1) & (df.alpha == 0.10) & (df.model == "lstm")]
    rows = []
    for a in ASSETS:
        blocos = []
        for n in NORMS:
            cel = []
            for c in ("lsc", "aci_raw", "aci_lsc"):
                r = d[(d.symbol == a) & (d.norm == n) & (d.construction == c)].iloc[0]
                if c == "lsc":
                    cel.append(f"{f3(r.coverage)} & {pct(r.rel_width)} & {f3(r.gap)}")
                else:
                    cel.append(f"{f3(r.coverage)} & {pct(r.rel_width)} & {int(r.n_inf)} & {f3(r.gap)}")
            blocos.append(f"& {NORM_LABEL[n]} & " + " & ".join(cel) + " \\\\")
        blocos[0] = f"\\multirow{{{len(blocos)}}}{{*}}{{{ASSET_LABEL[a]}}} " + blocos[0]
        rows += blocos + ["\\midrule"]
    rows = rows[:-1]
    tabela(rows, "Asset & Normalization & \\multicolumn{3}{c}{Locally scaled} & \\multicolumn{4}{c}{ACI (raw)} & \\multicolumn{4}{c}{ACI (scaled)} \\\\\n"
                 "\\cmidrule(lr){3-5}\\cmidrule(lr){6-9}\\cmidrule(lr){10-13} & & Cov. & Width & Gap & Cov. & Width & $\\infty$ h & Gap & Cov. & Width & $\\infty$ h & Gap \\\\",
           "Adaptive conformal inference ($\\gamma = 0.005$) on the raw and on the locally scaled score, LSTM backbone, $H = 1$, $\\alpha = 0.10$. "
           "Width in percent of price over the hours with a finite interval; ``$\\infty$ h'' is the number of test hours in which $\\alpha_t \\leq 0$ and the interval is the whole line.",
           "tab:aci", out, "llrrrrrrrrrrr", wide=True)


def tab5(df, out):
    """Persistencia vs melhor modelo treinado (menor IS no lsc), por ativo e horizonte."""
    d = ultimo_passo(df[(df.alpha == 0.10) & (df.construction == "lsc")])
    rows = []
    for a in ASSETS:
        blocos = []
        for H in (1, 6, 24):
            p = d[(d.symbol == a) & (d.H == H) & (d.model == "persistence")].iloc[0]
            m = d[(d.symbol == a) & (d.H == H) & (d.model != "persistence")]
            b = m.loc[m.is_rel.idxmin()]
            blocos.append(f"& {H} & Persistence & {f3(p.coverage)} & {pct(p.rel_width)} & {pct(p.is_rel)} & {f3(p.gap)} \\\\")
            blocos.append(f"& & {MODEL_LABEL[b.model]}, {NORM_LABEL[b.norm].lower()} & {f3(b.coverage)} & {pct(b.rel_width)} & {pct(b.is_rel)} & {f3(b.gap)} \\\\")
        blocos[0] = f"\\multirow{{{len(blocos)}}}{{*}}{{{ASSET_LABEL[a]}}} " + blocos[0]
        rows += blocos + ["\\midrule"]
    rows = rows[:-1]
    tabela(rows, "Asset & $H$ & Forecaster & Coverage & Width (\\%) & IS (\\%) & Gap \\\\",
           "Locally scaled interval around the persistence forecast against the trained forecaster with the lowest interval score "
           "for the same asset and horizon, last step $h = H$, $\\alpha = 0.10$.", "tab:persistence", out, "lllrrrr")


def tab6(df, out):
    """Gap medio por backbone x construcao (ancorada, H=1, alpha=0.10, media nos ativos)."""
    d = df[(df.H == 1) & (df.alpha == 0.10) & (df.norm == "anchored") & (df.model.isin(MODELS))]
    piv_gap = d.pivot_table(index="model", columns="construction", values="gap", aggfunc="mean")
    piv_cov = d.pivot_table(index="model", columns="construction", values="coverage", aggfunc="mean")
    rows = []
    for m in MODELS:
        rows.append(f"{MODEL_LABEL[m]} & " + " & ".join(f"{f3(piv_cov.loc[m, c])} & {f3(piv_gap.loc[m, c])}" for c in CONS) + " \\\\")
    tabela(rows, "Backbone & \\multicolumn{2}{c}{Split (raw)} & \\multicolumn{2}{c}{Locally scaled} & \\multicolumn{2}{c}{ACI (raw)} & \\multicolumn{2}{c}{ACI (scaled)} \\\\\n"
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9} & Cov. & Gap & Cov. & Gap & Cov. & Gap & Cov. & Gap \\\\",
           "Coverage and coverage gap averaged over the four assets, by backbone and interval construction, anchored normalization, $H = 1$, $\\alpha = 0.10$.",
           "tab:backbones", out, "lrrrrrrrr")
    return piv_gap, piv_cov


def fig1(df, out):
    """Cobertura por tercil, um painel por construcao, uma linha por normalizacao (media nos ativos, LSTM, H=1, 0.10)."""
    d = df[(df.H == 1) & (df.alpha == 0.10) & (df.model == "lstm")]
    fig, axes = plt.subplots(1, 4, figsize=(9, 2.6), sharey=True)
    for ax, c in zip(axes, CONS):
        for n in NORMS:
            r = d[(d.norm == n) & (d.construction == c)]
            y = [r[f"cov_T{k}"].mean() for k in (1, 2, 3)]
            ax.plot([1, 2, 3], y, "-o", color=COLORS[n], lw=2, ms=5, label=NORM_LABEL[n])
        ax.axhline(0.90, color="#555555", lw=1, ls="--")
        ax.set_title(CON_LABEL[c], fontsize=9)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["low", "mid", "high"])
        ax.set_xlabel("volatility tercile")
    axes[0].set_ylabel("empirical coverage")
    axes[0].set_ylim(0.4, 1.0)
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig2(df, out):
    """Largura media vs h para a ancorada (lsc, LSTM, grade H=24, 0.10), com referencia sqrt(h)."""
    d = df[(df.H == 24) & (df.alpha == 0.10) & (df.model == "lstm") & (df.norm == "anchored") & (df.construction == "lsc")]
    fig, ax = plt.subplots(figsize=(4.2, 3))
    h = np.arange(1, 25)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    for a, col in zip(ASSETS, cols):
        r = d[d.symbol == a].sort_values("step")
        ax.plot(r.step, 100 * r.rel_width, "-", color=col, lw=2, label=ASSET_LABEL[a])
        w1 = 100 * r[r.step == 1].rel_width.iloc[0]
        ax.plot(h, w1 * np.sqrt(h), ":", color=col, lw=1)
    ax.set_xlabel("horizon step $h$ (hours)")
    ax.set_ylabel("mean width (% of price)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 24])
    ax.set_xticklabels([1, 2, 4, 8, 16, 24])
    ax.legend(fontsize=8, title="solid: observed; dotted: $w_1\\sqrt{h}$", title_fontsize=7)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig3(npz_path, out, key="H1_BTCUSDT_lstm_anchored_a0.1"):
    """Trajetoria de alpha_t do ACI sobre o score bruto e o escalado."""
    z = np.load(npz_path, allow_pickle=True)
    ts = pd.to_datetime(z[key + "__timestamp"])
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    ax.plot(ts, z[key + "__aci_raw"], color=COLORS["aci_raw"], lw=1.2, label="ACI on raw score")
    ax.plot(ts, z[key + "__aci_lsc"], color=COLORS["aci_lsc"], lw=1.2, label="ACI on locally scaled score")
    ax.axhline(0.10, color="#555555", lw=1, ls="--")
    ax.axhline(0.0, color="#999999", lw=0.8)
    ax.set_ylabel("$\\alpha_t$")
    ax.legend(fontsize=8, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def manchetes(df):
    d1 = df[(df.H == 1) & (df.alpha == 0.10)]
    print("\n== manchetes ==")
    for m in MODELS:
        for c in CONS:
            for n in NORMS:
                r = d1[(d1.model == m) & (d1.norm == n) & (d1.construction == c)]
                print(f"H1 {m:12s} {n:9s} {c:8s} cov {r.coverage.mean():.3f} [{r.coverage.min():.3f},{r.coverage.max():.3f}] "
                      f"width {100 * r.rel_width.mean():.2f}% gap {r.gap.mean():+.3f} [{r.gap.min():+.3f},{r.gap.max():+.3f}] inf_h {r.n_inf.mean():.0f}")
    print("-- split (raw) por horizonte, LSTM anchored, ultimo passo:")
    d = ultimo_passo(df[(df.alpha == 0.10) & (df.model == "lstm") & (df.norm == "anchored")])
    for H in (1, 6, 24):
        for c in ("split", "lsc"):
            r = d[(d.H == H) & (d.construction == c)]
            print(f"   H={H:2d} {c:6s} cov {r.coverage.mean():.3f} [{r.coverage.min():.3f},{r.coverage.max():.3f}] width {100 * r.rel_width.mean():.2f}% gap {r.gap.mean():+.3f}")
    print("-- persistencia vs melhor modelo (lsc, 0.10, ultimo passo):")
    d = ultimo_passo(df[(df.alpha == 0.10) & (df.construction == "lsc")])
    for H in (1, 6, 24):
        for a in ASSETS:
            p = d[(d.symbol == a) & (d.H == H) & (d.model == "persistence")].iloc[0]
            m = d[(d.symbol == a) & (d.H == H) & (d.model != "persistence")]
            b = m.loc[m.is_rel.idxmin()]
            print(f"   H={H:2d} {ASSET_LABEL[a]} persist IS {100 * p.is_rel:.3f} width {100 * p.rel_width:.3f} | best {b.model}/{b.norm} IS {100 * b.is_rel:.3f} width {100 * b.rel_width:.3f}")
    print("-- sqrt(h): razao largura(h=24)/largura(h=1) na ancorada lsc LSTM (esperado 4.90):")
    d = df[(df.H == 24) & (df.alpha == 0.10) & (df.model == "lstm") & (df.norm == "anchored") & (df.construction == "lsc")]
    for a in ASSETS:
        r = d[d.symbol == a]
        print(f"   {ASSET_LABEL[a]} {r[r.step == 24].rel_width.iloc[0] / r[r.step == 1].rel_width.iloc[0]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.read_csv(os.path.join(a.res, "resultados.csv"))
    tdir, fdir = os.path.join(a.out, "tables"), os.path.join(a.out, "figures")
    os.makedirs(tdir, exist_ok=True)
    os.makedirs(fdir, exist_ok=True)
    tab1(df, os.path.join(tdir, "tab1_h1_lstm.tex"))
    tab1(df, os.path.join(tdir, "tabA1_h1_transformer.tex"), model="transformer", nome="tabA1")
    tab1(df, os.path.join(tdir, "tabA2_h1_linear.tex"), model="linear", nome="tabA2")
    tab2(df, os.path.join(tdir, "tab2_terciles.tex"))
    tab3(df, os.path.join(tdir, "tab3_horizons.tex"))
    tab4(df, os.path.join(tdir, "tab4_aci.tex"))
    tab5(df, os.path.join(tdir, "tab5_persistence.tex"))
    tab6(df, os.path.join(tdir, "tab6_backbones.tex"))
    fig1(df, os.path.join(fdir, "fig1_coverage_terciles.pdf"))
    fig2(df, os.path.join(fdir, "fig2_width_vs_h.pdf"))
    fig3(os.path.join(a.res, "aci_alpha_paths.npz"), os.path.join(fdir, "fig3_aci_alpha.pdf"))
    manchetes(df)
    print("ok:", tdir, fdir)


if __name__ == "__main__":
    main()
