#!/usr/bin/env python
"""
Artigo 2 — tabelas (.tex) e figuras (.pdf) a partir de results/artigo2/{resultados,D}.csv
(gerados por analysis/artigo2_degenerescencia.py).

    python analysis/artigo2_tabelas.py --res results/artigo2 --out <pasta do artigo no Overleaf>

Escreve <out>/tables/*.tex e <out>/figures/*.pdf e imprime as manchetes usadas no texto.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402

ASSET_LABEL = {"BTCUSDT": "BTC", "ETHUSDT": "ETH"}
POL_LABEL = {"legacy": "\\textsc{legacy}", "window_stats": "\\textsc{window-stats}"}
PRED_LABEL = {"linear": "linear", "mlp": "MLP"}
MODEL_LABEL = {"lstm": "LSTM", "transformer": "Transformer"}
# Paleta categorica validada (dataviz), ordem fixa por serie
COLORS = {"legacy_linear": "#2a78d6", "legacy_mlp": "#1baf7a", "window_stats": "#eb6834", "unaffected": "#eda100"}
MARK_H = {1: "o", 6: "s", 24: "^"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
                     "legend.frameon": False, "figure.dpi": 150})


def sci(x, zero="0"):
    """1 algarismo x 10^e; 0 exato vira `zero`."""
    if x is None or not np.isfinite(x):
        return "--"
    if x == 0:
        return zero
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(m, 1) >= 10:          # 9.96 -> "10.0e-10" vira 1.0e-9
        m, e = m / 10, e + 1
    return f"${m:.1f}{{\\times}}10^{{{e}}}$"


def f1(x):
    return f"{x:.1f}"


def f3(x):
    return f"{x:.3f}"


def tabela(rows, header, caption, label, out, col_fmt, extra_note="", wide=False, pos="t"):
    if not header.rstrip().endswith("\\\\"):
        header = header.rstrip() + " \\\\"
    abre = ["\\resizebox{\\textwidth}{!}{%"] if wide else []
    fecha = ["}"] if wide else []
    linhas = [f"\\begin{{table}}[{pos}]", "\\centering", "\\footnotesize", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
              *abre, f"\\begin{{tabular}}{{{col_fmt}}}", "\\toprule", header, "\\midrule", *rows, "\\bottomrule",
              "\\end{tabular}", *fecha]
    if extra_note:
        linhas.append(f"\\par\\smallskip\\raggedright\\footnotesize {extra_note}")
    linhas.append("\\end{table}")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")


def periodos(df, K):
    return [int(p) for p in df[df.K == K].iloc[0]["periods"].split("/")]


# ---------------------------------------------------------------- tabelas
def tab1(df, out):
    """Periodos selecionados por K e escalas afetadas (H < p) em cada horizonte."""
    rows = []
    for K in (2, 3, 4):
        ps = periodos(df, K)
        afet = [str(sum(p > H for p in ps)) for H in (1, 6, 24)]
        rows.append(f"{K} & {', '.join(str(p) for p in ps)} & {' & '.join(afet)} \\\\")
    tabela(rows, "$K$ & Selected periods $p_k$ (hours) & $H=1$ & $H=6$ & $H=24$",
           "Periods selected by the spectral criterion on standardized channels with $L = 96$ (identical for BTC/USDT and ETH/USDT) "
           "and number of scales with $H < p_k$ at each horizon.",
           "tab:periods", os.path.join(out, "tables", "tab1_periods.tex"), "clccc",
           extra_note="The criterion returns the harmonics of the window, $96/k$ for $k = 1, \\ldots, K$; the only scale with $H \\geq p$ in the grid is $p = 24$ at $H = 24$ ($K = 4$).")


def tab2(df, dD, out):
    """Degenerescencia por H, K e ativo: std do alvo normalizado (legacy), r, D."""
    rows = []
    for H in (1, 6, 24):
        for K in (2, 3, 4):
            for sym in ("BTCUSDT", "ETHUSDT"):
                sel = df[(df.H == H) & (df.K == K) & (df.symbol == sym) & (df.model == "lstm") & (df.predictor == "linear")]
                leg = sel[sel.policy == "legacy"].iloc[0]
                ws = sel[sel.policy == "window_stats"].iloc[0]
                d = dD[(dD.H == H) & (dD.K == K) & (dD.symbol == sym)]
                d_leg_lin = d[(d.policy == "legacy") & (d.predictor == "linear")].iloc[0]["D"]
                d_leg_mlp = d[(d.policy == "legacy") & (d.predictor == "mlp")].iloc[0]["D"]
                d_ws = d[(d.policy == "window_stats") & (d.predictor == "linear")].iloc[0]["D"]
                std_leg = "/".join(f"{float(s):.2g}" if float(s) >= 0.01 else ("0" if float(s) == 0 else f"{float(s):.0e}".replace("e-0", "e-")) for s in leg["target_std"].split("/"))
                std_ws = "/".join(f"{float(s):.2g}" for s in ws["target_std"].split("/"))
                rows.append(f"{H} & {K} & {ASSET_LABEL[sym]} & {leg['n_affected']}/{K} & {std_leg} & {std_ws} & "
                            f"{sci(leg['r_ens'])} & {f3(ws['r_ens'])} & {sci(d_leg_lin, zero='$0$')} & {sci(d_leg_mlp)} & {sci(d_ws)} \\\\")
            if (H, K) != (24, 4):
                rows.append("\\addlinespace[1pt]")
        if H != 24:
            rows.append("\\midrule")
    header = ("$H$ & $K$ & Asset & Affected & \\multicolumn{2}{c}{$\\operatorname{std}(\\tilde{y})$ per scale} & \\multicolumn{2}{c}{$r$ (LSTM)} & \\multicolumn{3}{c}{$D$} \\\\\n"
              "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-11}\n"
              " & & & scales & \\textsc{legacy} & \\textsc{w-stats} & \\textsc{legacy} & \\textsc{w-stats} & \\textsc{legacy}, lin. & \\textsc{legacy}, MLP & \\textsc{w-stats}")
    tabela(rows, header,
           "Degeneracy across horizons and scales. Affected scales are those with $H < p_k$. $\\operatorname{std}(\\tilde{y})$ is the standard deviation of the normalized training target per scale (periods in the order of Table~\\ref{tab:periods}); $r$ is the contribution ratio \\eqref{eq:r} of the ensemble forecast of the LSTM (linear predictor); $D$ is the backbone invariance statistic \\eqref{eq:D} between the LSTM and the Transformer trained under the same normalization (\\textsc{window-stats} is identical for both predictors wherever every scale is affected).",
           "tab:degeneracy", os.path.join(out, "tables", "tab2_degeneracy.tex"), "cclcllllrrr", wide=True,
           extra_note="$D = 0$ means that the two prediction files are identical to the last bit. Under \\textsc{legacy} at $H = 1$ the normalized target is zero up to floating point error and $r$ is at the level of $\\epsilon^2 = 10^{-16}$ times the variance ratio; at $H = 6$ and $H = 24$ the target is not null but the predicted dispersion is a fraction of the true one, and $r$ and $D$ are one to two orders of magnitude below their \\textsc{window-stats} values.")


def tab3(df, out, K=3, nome="tab3_correction_k3", appendix=False):
    rows = []
    for H in (1, 6, 24):
        for sym in ("BTCUSDT", "ETHUSDT"):
            for pol in ("legacy", "window_stats"):
                for pred in ("linear", "mlp"):
                    sel = df[(df.H == H) & (df.K == K) & (df.symbol == sym) & (df.policy == pol) & (df.predictor == pred)]
                    l = sel[sel.model == "lstm"].iloc[0]
                    t = sel[sel.model == "transformer"].iloc[0]
                    rows.append(f"{H} & {ASSET_LABEL[sym]} & {POL_LABEL[pol]} & {PRED_LABEL[pred]} & "
                                f"{l['rmse_persist']:.1f} & {l['rmse_test']:.1f} & {t['rmse_test']:.1f} & "
                                f"{f3(l['ratio_persist'])} & {f3(t['ratio_persist'])} & "
                                f"{l['mae_test']:.1f} & {t['mae_test']:.1f} & {f1(l['dm_stat'])} & {f1(t['dm_stat'])} \\\\")
            rows.append("\\addlinespace[1pt]")
        if H != 24:
            rows.append("\\midrule")
    header = ("$H$ & Asset & Rule & Pred. & Persist. & \\multicolumn{2}{c}{RMSE} & \\multicolumn{2}{c}{RMSE / persist.} & \\multicolumn{2}{c}{MAE} & \\multicolumn{2}{c}{DM vs persist.} \\\\\n"
              "\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}\\cmidrule(lr){12-13}\n"
              " & & & & RMSE & LSTM & Transf. & LSTM & Transf. & LSTM & Transf. & LSTM & Transf.")
    tabela(rows, header,
           f"Effect of the correction, $K = {K}$. Test RMSE and MAE in USDT, ratio of the RMSE to the persistence forecast, and Diebold-Mariano statistic against persistence (positive values favour persistence; Newey-West variance, lag 10, Harvey correction). Under \\textsc{{legacy}} the two backbones give the same numbers; under \\textsc{{window-stats}} the two predictors give the same numbers wherever every scale is affected, because the auxiliary predictor is not used.",
           "tab:correction" if not appendix else f"tab:correction_k{K}", os.path.join(out, "tables", f"{nome}.tex"), "clllrrrrrrrrr", wide=True)


def tabA1(df, out):
    """Razao RMSE/persistencia para todos os K."""
    rows = []
    for H in (1, 6, 24):
        for sym in ("BTCUSDT", "ETHUSDT"):
            for pol in ("legacy", "window_stats"):
                for pred in ("linear", "mlp"):
                    cels = []
                    for K in (2, 3, 4):
                        sel = df[(df.H == H) & (df.K == K) & (df.symbol == sym) & (df.policy == pol) & (df.predictor == pred)]
                        cels += [f3(sel[sel.model == "lstm"].iloc[0]["ratio_persist"]), f3(sel[sel.model == "transformer"].iloc[0]["ratio_persist"])]
                    rows.append(f"{H} & {ASSET_LABEL[sym]} & {POL_LABEL[pol]} & {PRED_LABEL[pred]} & {' & '.join(cels)} \\\\")
            rows.append("\\addlinespace[1pt]")
        if H != 24:
            rows.append("\\midrule")
    header = ("$H$ & Asset & Rule & Pred. & \\multicolumn{2}{c}{$K = 2$} & \\multicolumn{2}{c}{$K = 3$} & \\multicolumn{2}{c}{$K = 4$} \\\\\n"
              "\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}\n"
              " & & & & LSTM & Transf. & LSTM & Transf. & LSTM & Transf.")
    tabela(rows, header, "Ratio of the test RMSE to the persistence forecast for every number of scales $K$.",
           "tab:allK", os.path.join(out, "tables", "tabA1_ratio_allK.tex"), "cllllrrrrr", wide=True, pos="!htb")


# ---------------------------------------------------------------- figuras
def fig1(df, dD, out):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    # (a) r por escala vs p/H
    pts = {"legacy_aff": [], "legacy_unaff": [], "window_stats": []}
    for _, row in df[df.model == "lstm"].iterrows():
        ps = [int(p) for p in row["periods"].split("/")]
        rs = [float(x) for x in row["r_scales"].split("/")]
        for p, r in zip(ps, rs):
            key = "window_stats" if row["policy"] == "window_stats" else ("legacy_aff" if p > row["H"] else "legacy_unaff")
            pts[key].append((p / row["H"], max(r, 1e-36), row["H"]))
    for key, cor, lab in (("legacy_aff", COLORS["legacy_linear"], "\\textsc{legacy}, $H<p$"),
                          ("legacy_unaff", COLORS["unaffected"], "\\textsc{legacy}, $H = p$"),
                          ("window_stats", COLORS["window_stats"], "\\textsc{window-stats}")):
        arr = np.array(pts[key])
        for H in (1, 6, 24):
            m = arr[:, 2] == H
            if m.any():
                ax.scatter(arr[m, 0], arr[m, 1], s=22, marker=MARK_H[H], facecolors=cor, edgecolors="white", linewidths=0.6, alpha=0.9,
                           label=lab.replace("\\textsc{", "").replace("}", "") if H == (24 if key == "legacy_unaff" else 1) else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$p / H$"); ax.set_ylabel("contribution ratio $r$ (per scale)")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 96]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "96"])
    ax.set_ylim(1e-36, 1)
    ax.annotate("$H = 1$", (50, 3e-33), fontsize=8, color="#555555")
    ax.annotate("$H = 6$", (5.5, 3e-10), fontsize=8, color="#555555")
    ax.annotate("$H = 24$", (0.95, 3e-10), fontsize=8, color="#555555")
    ax.legend(loc="center left", fontsize=7.5, handletextpad=0.2)
    ax.set_title("(a) backbone share of the forecast variance", fontsize=9, loc="left")
    # (b) D vs H
    for pol, cor in (("legacy", COLORS["legacy_linear"]), ("window_stats", COLORS["window_stats"])):
        for pred, ls in (("linear", "-"), ("mlp", "--")):
            g = dD[(dD.policy == pol) & (dD.predictor == pred)].groupby("H")["D"].agg(["min", "max", "mean"]).reindex([1, 6, 24])
            y = g["mean"].clip(lower=1e-20)
            bx.plot([1, 6, 24], y, ls, color=cor, lw=2, marker="o", ms=4,
                    label=f"{'legacy' if pol == 'legacy' else 'window-stats'}, {PRED_LABEL[pred]}")
            bx.fill_between([1, 6, 24], g["min"].clip(lower=1e-20), g["max"].clip(lower=1e-20), color=cor, alpha=0.15, lw=0)
    bx.set_yscale("log"); bx.set_xscale("log"); bx.set_xticks([1, 6, 24]); bx.set_xticklabels(["1", "6", "24"])
    bx.set_ylim(1e-20, 1e-1); bx.set_xlabel("horizon $H$ (hours)"); bx.set_ylabel("backbone invariance $D$")
    bx.annotate("exact zeros drawn at $10^{-20}$", (1.05, 1.6e-20), fontsize=7, color="#555555")
    bx.legend(fontsize=7.5, loc="center right")
    bx.set_title("(b) LSTM vs Transformer disagreement", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "fig1_r_and_D.pdf")); plt.close(fig)


def fig2(df, out, K=3):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    series = (("legacy", "linear", COLORS["legacy_linear"], "legacy, linear"), ("legacy", "mlp", COLORS["legacy_mlp"], "legacy, MLP"),
              ("window_stats", "linear", COLORS["window_stats"], "window-stats"))
    for ax, sym in zip(axes, ("BTCUSDT", "ETHUSDT")):
        for pol, pred, cor, lab in series:
            for model, ls, mk in (("lstm", "-", "o"), ("transformer", "--", "s")):
                sel = df[(df.K == K) & (df.symbol == sym) & (df.policy == pol) & (df.predictor == pred) & (df.model == model)].sort_values("H")
                ax.plot(sel["H"], sel["ratio_persist"], ls, color=cor, lw=2, marker=mk, ms=4, mfc=cor if model == "lstm" else "white",
                        label=f"{lab} ({MODEL_LABEL[model]})" if sym == "BTCUSDT" else None)
        ax.axhline(1.0, color="#888888", lw=1, ls=":")
        ax.set_xscale("log"); ax.set_xticks([1, 6, 24]); ax.set_xticklabels(["1", "6", "24"])
        ax.set_xlabel("horizon $H$ (hours)"); ax.set_title(ASSET_LABEL[sym] + "/USDT", fontsize=9, loc="left")
    axes[0].set_ylabel("RMSE / persistence RMSE")
    axes[0].legend(fontsize=7, ncol=1, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "fig2_ratio_vs_H.pdf")); plt.close(fig)


def manchetes(df, dD):
    print("\n== MANCHETES ==")
    for H in (1, 6, 24):
        for pol in ("legacy", "window_stats"):
            s = df[(df.H == H) & (df.policy == pol)]
            d = dD[(dD.H == H) & (dD.policy == pol)]
            print(f"H={H:2d} {pol:12s} ratio {s.ratio_persist.min():.3f}-{s.ratio_persist.max():.3f} | "
                  f"linear {s[s.predictor=='linear'].ratio_persist.min():.3f}-{s[s.predictor=='linear'].ratio_persist.max():.3f} "
                  f"mlp {s[s.predictor=='mlp'].ratio_persist.min():.3f}-{s[s.predictor=='mlp'].ratio_persist.max():.3f} | "
                  f"r_ens {s.r_ens.min():.1e}-{s.r_ens.max():.1e} | D {d.D.min():.1e}-{d.D.max():.1e} | DM {s.dm_stat.min():.1f}-{s.dm_stat.max():.1f} | "
                  f"bet {s.bet_ratio.min():.2f}-{s.bet_ratio.max():.2f} | std_min {s.target_std_min.min():.2g}-{s.target_std_min.max():.2g}")
    # melhora relativa do RMSE
    for H in (1, 6, 24):
        for pred in ("linear", "mlp"):
            a = df[(df.H == H) & (df.policy == "legacy") & (df.predictor == pred)].set_index(["symbol", "K", "model"])["rmse_test"]
            b = df[(df.H == H) & (df.policy == "window_stats") & (df.predictor == pred)].set_index(["symbol", "K", "model"])["rmse_test"]
            red = 1 - b / a
            print(f"H={H:2d} {pred:6s} reducao RMSE window_stats vs legacy: {red.min():.1%}-{red.max():.1%}")
    # diferenca entre backbones sob window_stats
    for H in (1, 6, 24):
        s = df[(df.H == H) & (df.policy == "window_stats") & (df.predictor == "linear")]
        piv = s.pivot_table(index=["symbol", "K"], columns="model", values="ratio_persist")
        print(f"H={H:2d} window_stats |LSTM-Transf| ratio: {np.abs(piv.lstm - piv.transformer).min():.3f}-{np.abs(piv.lstm - piv.transformer).max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    df = pd.read_csv(os.path.join(a.res, "resultados.csv"))
    dD = pd.read_csv(os.path.join(a.res, "D.csv"))
    tab1(df, a.out); tab2(df, dD, a.out); tab3(df, a.out); tabA1(df, a.out)
    fig1(df, dD, a.out); fig2(df, a.out)
    manchetes(df, dD)
    print("tabelas:", sorted(os.listdir(os.path.join(a.out, "tables"))))
    print("figuras:", sorted(os.listdir(os.path.join(a.out, "figures"))))


if __name__ == "__main__":
    main()
