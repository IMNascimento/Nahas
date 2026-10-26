#!/usr/bin/env python
"""
Artigo 1 — tabelas (.tex) e figuras (.pdf) a partir de results/artigo1/*.csv
(gerados por analysis/artigo1_resultados.py).

    python analysis/artigo1_tabelas.py --res results/artigo1 --out <pasta do artigo no Overleaf>

Escreve <out>/tables/*.tex e <out>/figures/*.pdf e imprime as manchetes usadas no texto.
As tabelas de origem rolante e de varredura de L so sao escritas se os CSVs tiverem dados.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402
import pandas as pd               # noqa: E402
from scipy import stats           # noqa: E402

ASSET = {"BTCUSDT": "BTC", "ETHUSDT": "ETH", "SOLUSDT": "SOL", "BNBUSDT": "BNB"}
ASSETS = list(ASSET)
NORM = {"global": "Global Z", "winZ": "Window Z", "winMM": "Window MinMax", "anchored": "Anchored", "evomsn": "EvoMSN"}
NORMS = ["global", "winZ", "winMM", "anchored", "evomsn"]
MODEL = {"lstm": "LSTM", "transformer": "Transformer", "linear": "Linear"}
MODELS = ["lstm", "transformer", "linear"]
SEED_PRINCIPAL = 1482563973
# paleta categorica validada (dataviz), ordem fixa por normalizacao
COLORS = {"global": "#2a78d6", "winZ": "#1baf7a", "winMM": "#eb6834", "anchored": "#eda100", "evomsn": "#7a5cc7"}
MARK = {"lstm": "o", "transformer": "s", "linear": "^"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#dddddd", "grid.linewidth": 0.5,
                     "legend.frameon": False, "figure.dpi": 150})


def sci(x):
    if x is None or not np.isfinite(x):
        return "--"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(m, 2) >= 10:
        m, e = m / 10, e + 1
    return f"${m:.2f}{{\\times}}10^{{{e}}}$"


def fnum(x, nd=3):
    return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def pm(m, s, nd=3):
    return f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


def tabela(rows, header, caption, label, out, col_fmt, extra_note="", wide=False, pos="t", size="footnotesize"):
    if not header.rstrip().endswith("\\\\"):
        header = header.rstrip() + " \\\\"
    abre = ["\\resizebox{\\textwidth}{!}{%"] if wide else []
    fecha = ["}"] if wide else []
    linhas = [f"\\begin{{table}}[{pos}]", "\\centering", f"\\{size}", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
              *abre, f"\\begin{{tabular}}{{{col_fmt}}}", "\\toprule", header, "\\midrule", *rows, "\\bottomrule",
              "\\end{tabular}", *fecha]
    if extra_note:
        linhas.append(f"\\par\\smallskip\\raggedright\\footnotesize {extra_note}")
    linhas.append("\\end{table}")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas) + "\n")


# ---------------------------------------------------------------- tabelas
def tab_zero(z, out):
    rows = []
    for sym in ASSETS:
        for i, n in enumerate(NORMS):
            r = z[(z.symbol == sym) & (z.norm == n)]
            if r.empty:
                continue
            r = r.iloc[0]
            first = f"\\multirow{{{len(NORMS)}}}{{*}}{{{ASSET[sym]}}}" if i == 0 else ""
            zero = fnum(r.rmse_zero, 1) if np.isfinite(r.rmse_zero) else "n/a"
            rows.append(f"{first} & {NORM[n]} & {zero} & {fnum(r.rmse_persist, 1)} & {fnum(r.rmse_lstm, 1)} & "
                        f"{fnum(r.rmse_transformer, 1)} & {fnum(r.rmse_linear, 1)} \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    tabela(rows, "Asset & Normalization & Zero output & Persistence & LSTM & Transformer & Linear",
           "The implicit forecast. RMSE, in the original units, of the forecast issued at zero network output (Proposition~\\ref{prop:implicit}), "
           "of persistence and of the trained networks (mean over five seeds), $H = 1$, $L = 96$. The zero-output forecast is the training mean under "
           "global Z-score, the window mean under window Z-score, the window minimum under window MinMax and the last price under the anchored operator, "
           "where it coincides with persistence by construction. Under EvoMSN the zero output returns the multi-scale backbone forecast, which is not a "
           "closed-form statistic of the window and is analysed in the companion paper.",
           "tab:zero", out, "llrrrrr")


def tab_amp(a, out):
    rows = []
    a = a[a.model == "lstm"]
    for sym in ASSETS:
        for i, n in enumerate(NORMS):
            r = a[(a.symbol == sym) & (a.norm == n)]
            if r.empty:
                continue
            r = r.iloc[0]
            first = f"\\multirow{{{len(NORMS)}}}{{*}}{{{ASSET[sym]}}}" if i == 0 else ""
            rows.append(f"{first} & {NORM[n]} & {sci(r.mean_s2)} & {sci(r.mean_eps2)} & {sci(r['product'])} & {sci(r.mse)} & "
                        f"{fnum(r.cov_term / r.mse, 3)} & {fnum(r.corr_s2_eps2, 3)} \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    tabela(rows, "Asset & Normalization & $\\overline{s^2}$ & $\\overline{\\varepsilon^2}$ & $\\overline{s^2}\\,\\overline{\\varepsilon^2}$ & MSE & "
           "$\\mathrm{Cov}/\\mathrm{MSE}$ & $\\rho(s^2, \\varepsilon^2)$",
           "Amplification (Proposition~\\ref{prop:amplification}), LSTM backbone, $H = 1$, seed of the main grid. Mean squared scale over the test set, "
           "mean normalized squared error, their product, the realized MSE in the original units, the covariance term $\\mathrm{MSE} - \\overline{s^2}\\,\\overline{\\varepsilon^2}$ "
           "as a share of the MSE, and the correlation between $s^2$ and $\\varepsilon^2$. Under the global scaler $s$ is constant and the covariance is exactly zero.",
           "tab:amp", out, "llrrrrrr")


def _cell(g, col, nd):
    return pm(g[col].mean(), g[col].std(ddof=1), nd)


def tab_main(h1, out_main, out_app):
    rows, rows_app = [], []
    for sym in ASSETS:
        for i, n in enumerate(NORMS):
            first = f"\\multirow{{{len(NORMS)}}}{{*}}{{{ASSET[sym]}}}" if i == 0 else ""
            cells, cells_app = [], []
            for m in MODELS:
                g = h1[(h1.symbol == sym) & (h1.norm == n) & (h1.model == m)]
                cells.append(f"{_cell(g, 'U', 3)} & {g.dm_stat.mean():.1f}")
                cells_app.append(f"{g.rmse.mean():.1f} & {g.mae.mean():.1f} & {g.corr_returns.mean():.3f} & {g.dir_acc.mean():.3f}")
            rows.append(f"{first} & {NORM[n]} & " + " & ".join(cells) + " \\\\")
            rows_app.append(f"{first} & {NORM[n]} & " + " & ".join(cells_app) + " \\\\")
        rows.append("\\midrule"); rows_app.append("\\midrule")
    rows, rows_app = rows[:-1], rows_app[:-1]
    head = ("\\multirow{2}{*}{Asset} & \\multirow{2}{*}{Normalization} & \\multicolumn{2}{c}{LSTM} & \\multicolumn{2}{c}{Transformer} & \\multicolumn{2}{c}{Linear} \\\\"
            "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8} & & $U$ & DM & $U$ & DM & $U$ & DM")
    tabela(rows, head,
           "Main comparison at $H = 1$, $L = 96$: Theil's $U$ (mean $\\pm$ one standard deviation over five seeds) and the Diebold-Mariano statistic against persistence "
           "(mean over seeds; positive means a larger error than persistence; $|\\mathrm{DM}| > 1.96$ is significant at the 5\\% level). "
           "RMSE, MAE, return correlation and directional accuracy are in Table~\\ref{tab:mainapp}.",
           "tab:main", out_main, "llrrrrrr")
    head_app = ("\\multirow{2}{*}{Asset} & \\multirow{2}{*}{Normalization} & \\multicolumn{4}{c}{LSTM} & \\multicolumn{4}{c}{Transformer} & \\multicolumn{4}{c}{Linear} \\\\"
                "\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\\cmidrule(lr){11-14} & & RMSE & MAE & $\\rho$ & DA & RMSE & MAE & $\\rho$ & DA & RMSE & MAE & $\\rho$ & DA")
    tabela(rows_app, head_app,
           "Main comparison at $H = 1$, $L = 96$, means over five seeds: RMSE and MAE in the original units (USDT), correlation $\\rho$ between forecast change and realized change, "
           "and directional accuracy DA. Persistence has $\\rho = 0$ and $\\mathrm{DA} = 0.5$ by construction.",
           "tab:mainapp", out_app, "llrrrrrrrrrrrr", wide=True, pos="!htb")


def tab_horizon(df, out):
    rows = []
    for m in MODELS:
        cells = []
        for n in ("anchored", "winZ", "global"):
            for H in (1, 6, 24):
                g = df[(df.H == H) & (df.model == m) & (df.norm == n)]
                if H == 1:
                    g = g[g.seed == SEED_PRINCIPAL]
                cells.append(f"{g.U.mean():.3f}")
        rows.append(f"{MODEL[m]} & " + " & ".join(cells) + " \\\\")
    head = ("\\multirow{2}{*}{Backbone} & \\multicolumn{3}{c}{Anchored} & \\multicolumn{3}{c}{Window Z} & \\multicolumn{3}{c}{Global Z} \\\\"
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10} & $H{=}1$ & $6$ & $24$ & $1$ & $6$ & $24$ & $1$ & $6$ & $24$")
    tabela(rows, head,
           "Theil's $U$ by horizon, averaged over the four assets, seed of the main grid, $L = 96$. For $H > 1$ the RMSE is computed over the $H$ steps of the forecast vector and persistence is $p_t$ at every step.",
           "tab:horizon", out, "lrrrrrrrrr")


def tab_terciles(t, out):
    rows = []
    for sym in ASSETS:
        for i, n in enumerate(["anchored", "winZ", "winMM", "global"]):
            first = f"\\multirow{{4}}{{*}}{{{ASSET[sym]}}}" if i == 0 else ""
            cells = []
            for m in ("lstm", "transformer"):
                r = t[(t.symbol == sym) & (t.norm == n) & (t.model == m)].iloc[0]
                cells.append(f"{r.U_T1:.3f} & {r.U_T2:.3f} & {r.U_T3:.3f}")
            rows.append(f"{first} & {NORM[n]} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    head = ("\\multirow{2}{*}{Asset} & \\multirow{2}{*}{Normalization} & \\multicolumn{3}{c}{LSTM} & \\multicolumn{3}{c}{Transformer} \\\\"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8} & & low & mid & high & low & mid & high")
    tabela(rows, head,
           "Theil's $U$ by tercile of causal realized volatility (standard deviation of the 24 hourly log returns ending at $t - 1$), $H = 1$, seed of the main grid. "
           "Persistence RMSE grows from the low to the high tercile by a factor of 1.5 to 2.1 depending on the asset.",
           "tab:terciles", out, "llrrrrrr")


def tab_seeds(h1, out):
    """t pareado sobre as 5 sementes: anchored vs winZ e vs winMM, em log RMSE (negativo = ancorada melhor)."""
    rows = []
    res = []
    for sym in ASSETS:
        for i, m in enumerate(MODELS):
            first = f"\\multirow{{3}}{{*}}{{{ASSET[sym]}}}" if i == 0 else ""
            cells = []
            for other in ("winZ", "winMM"):
                a = h1[(h1.symbol == sym) & (h1.model == m) & (h1.norm == "anchored")].sort_values("seed")
                b = h1[(h1.symbol == sym) & (h1.model == m) & (h1.norm == other)].sort_values("seed")
                assert list(a.seed) == list(b.seed) and len(a) == 5
                d = np.log(a.rmse.to_numpy()) - np.log(b.rmse.to_numpy())
                tt = stats.ttest_rel(np.log(a.rmse.to_numpy()), np.log(b.rmse.to_numpy()))
                cells.append(f"{100 * d.mean():.1f} & {tt.statistic:.1f} & {int((d < 0).sum())}/5")
                res.append({"symbol": sym, "model": m, "vs": other, "mean_log_diff": d.mean(), "t": tt.statistic, "p": tt.pvalue, "n_neg": int((d < 0).sum())})
            rows.append(f"{first} & {MODEL[m]} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    head = ("\\multirow{2}{*}{Asset} & \\multirow{2}{*}{Backbone} & \\multicolumn{3}{c}{Anchored vs.\\ window Z} & \\multicolumn{3}{c}{Anchored vs.\\ window MinMax} \\\\"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8} & & $\\Delta$ (\\%) & $t$ & sign")
    tabela(rows, head,
           "Paired comparison over five seeds, $H = 1$, $L = 96$: mean difference in log RMSE between the anchored operator and the window operator "
           "(in percent; negative means the anchored operator has the smaller error), paired $t$ statistic with four degrees of freedom "
           "($|t| > 2.78$ is significant at the 5\\% level) and number of seeds on which the anchored operator is better.",
           "tab:seeds", out, "llrrrrrr")
    return pd.DataFrame(res)


def tab_rolling(ro, out):
    if ro.empty:
        return None
    rows = []
    ro = ro.copy()
    ro["cfg"] = ro.model.map(MODEL) + " / " + ro.norm.map(NORM)
    for sym in ASSETS:
        sub = ro[ro.symbol == sym]
        if sub.empty:
            continue
        origins = sorted(sub.origin.unique())
        cfgs = sorted(sub.cfg.unique())
        # ranking por origem: persistencia (U=1) entra na ordenacao
        ranks = {}
        for o in origins:
            s = sub[sub.origin == o].set_index("cfg")["U"].to_dict()
            s["Persistence"] = 1.0
            order = sorted(s, key=s.get)
            ranks[o] = {c: (order.index(c) + 1, s[c]) for c in s}
        for i, c in enumerate(cfgs + ["Persistence"]):
            first = f"\\multirow{{{len(cfgs) + 1}}}{{*}}{{{ASSET[sym]}}}" if i == 0 else ""
            cells = [f"{ranks[o][c][1]:.3f} ({ranks[o][c][0]})" if c in ranks[o] else "--" for o in origins]
            rows.append(f"{first} & {c} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    n_o = len(sorted(ro.origin.unique()))
    head = "Asset & Configuration & " + " & ".join(f"Origin {o + 1}" for o in range(n_o))
    tabela(rows, head,
           "Rolling origin: Theil's $U$ and rank (in parentheses, 1 = smallest error) of each configuration in each of the five forecast origins, "
           "$H = 1$, $L = 96$, seed of the main grid. Each origin trains on an expanding window that ends at the origin and tests on the following 90 days; the test blocks do not overlap.",
           "tab:rolling", out, "ll" + "r" * n_o, wide=True)
    return ro


def tab_lsweep(ls, out):
    Ls = sorted(ls.L.unique())
    if len(Ls) < 2:
        return None
    rows = []
    for sym in ASSETS:
        for i, m in enumerate(("lstm", "transformer")):
            first = f"\\multirow{{2}}{{*}}{{{ASSET[sym]}}}" if i == 0 else ""
            cells = []
            for n in ("anchored", "winZ"):
                for L in Ls:
                    g = ls[(ls.symbol == sym) & (ls.model == m) & (ls.norm == n) & (ls.L == L)]
                    cells.append(pm(g.U.mean(), g.U.std(ddof=1), 3) if len(g) > 1 else (fnum(g.U.mean(), 3) if len(g) else "--"))
            rows.append(f"{first} & {MODEL[m]} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule")
    rows = rows[:-1]
    k = len(Ls)
    head = (f"\\multirow{{2}}{{*}}{{Asset}} & \\multirow{{2}}{{*}}{{Backbone}} & \\multicolumn{{{k}}}{{c}}{{Anchored}} & \\multicolumn{{{k}}}{{c}}{{Window Z}} \\\\"
            f"\\cmidrule(lr){{3-{2 + k}}}\\cmidrule(lr){{{3 + k}-{2 + 2 * k}}} & & " + " & ".join(f"$L{{=}}{L}$" for L in Ls) + " & " + " & ".join(f"${L}$" for L in Ls))
    tabela(rows, head,
           "Window length: Theil's $U$ (mean $\\pm$ one standard deviation over five seeds) for the anchored and the window Z-score operators, $H = 1$.",
           "tab:lsweep", out, "ll" + "r" * (2 * k), wide=True)
    return Ls


# ---------------------------------------------------------------- figuras
def fig_implicit(runs, out):
    """BTC: preco realizado no teste, previsao implicita global (media do treino) e ancorada (p_t)."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis.artigo1_resultados import precos, alinhar
    r = runs[(runs.H == 1) & (runs.seed == SEED_PRINCIPAL) & (runs.symbol == "BTCUSDT") & (runs.model == "lstm")]
    rg, ra = r[r.norm == "global"].iloc[0], r[r.norm == "anchored"].iloc[0]
    px = precos("BTCUSDT")
    te = alinhar(pd.read_csv(ra.test_csv), 1, px)
    tg = pd.read_csv(rg.test_csv)
    tr = px[(px.index >= pd.Timestamp(rg.train_first)) & (px.index <= pd.Timestamp(rg.train_last))]
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    t = pd.to_datetime(te["timestamp"])
    ax.plot(t, te["real_value"], color="#444444", lw=1.0, label="Realized price")
    ax.plot(t, te["p_t"], color=COLORS["anchored"], lw=1.0, ls="--", label="Implicit forecast, anchored ($p_t$)")
    ax.plot(t, tg["predicted_value"], color=COLORS["global"], lw=0.8, alpha=0.9, label="Trained LSTM, global Z")
    ax.axhline(tr.mean(), color=COLORS["global"], lw=1.4, ls=":", label=f"Implicit forecast, global Z (training mean, {tr.mean():,.0f})")
    ax.axhline(tr.max(), color="#999999", lw=0.8, ls=":", label=f"Training maximum ({tr.max():,.0f})")
    ax.set_ylabel("BTC/USDT")
    ax.legend(ncol=2, fontsize=7, loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return tr.mean(), tr.max()


def fig_U(h1, out):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.6), sharey=True)
    x = np.arange(len(NORMS))
    for ax, sym in zip(axes, ASSETS):
        for j, m in enumerate(MODELS):
            g = h1[(h1.symbol == sym) & (h1.model == m)].groupby("norm")["U"].agg(["mean", "std"]).reindex(NORMS)
            ax.errorbar(x + (j - 1) * 0.22, g["mean"], yerr=g["std"], fmt=MARK[m], ms=4, lw=0.8, capsize=2,
                        color=["#2a78d6", "#eb6834", "#1baf7a"][j], label=MODEL[m], mfc="white" if m == "transformer" else None)
        ax.axhline(1, color="#444444", lw=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.9, 12)
        ax.set_yticks([1, 1.1, 1.5, 2, 3, 5, 10])
        ax.set_yticklabels(["1", "1.1", "1.5", "2", "3", "5", "10"])
        ax.set_xticks(x)
        ax.set_xticklabels(["Glob. Z", "Win. Z", "Win. MM", "Anch.", "EvoMSN"], rotation=45, ha="right", fontsize=7)
        ax.set_title(ASSET[sym], fontsize=9)
    axes[0].set_ylabel("Theil's $U$ (log scale)")
    axes[0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_horizon(hs, out):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=True)
    for ax, sym in zip(axes, ASSETS):
        for m, c in (("lstm", "#2a78d6"), ("transformer", "#eb6834")):
            g = hs[(hs.symbol == sym) & (hs.model == m) & (hs.norm == "anchored")].sort_values("h")
            if g.empty:
                continue
            ref = g.rmse.iloc[0] * np.sqrt(g.h)
            ax.plot(g.h, g.rmse / ref, marker=MARK[m], ms=3, lw=1, color=c, label=f"{MODEL[m]}, anchored", mfc="white" if m == "transformer" else None)
            ax.plot(g.h, g.rmse_persist / ref, lw=0.8, ls="--", color="#444444", label="Persistence" if m == "lstm" else None)
        ax.axhline(1, color="#bbbbbb", lw=0.6)
        ax.set_title(ASSET[sym], fontsize=9)
        ax.set_xlabel("$h$ (hours)")
        ax.set_xticks([1, 6, 12, 18, 24])
    axes[0].set_ylabel("RMSE$(h)$ / $(\\sqrt{h}\\,$RMSE$(1))$")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_dispersion(d, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    x = np.arange(3)
    w = 0.2
    for ax, key, title in ((axes[0], "sd_anchored", "Anchored: $(y - p_t)/s(W)$"), (axes[1], "sd_rellast", "Relative-to-last: $(y - p_t)/p_t$")):
        for j, sym in enumerate(ASSETS):
            r = d[d.symbol == sym].iloc[0]
            vals = [r[f"{key}_T{k}"] for k in (1, 2, 3)]
            ax.bar(x + (j - 1.5) * w, vals, width=w * 0.9, color=["#2a78d6", "#1baf7a", "#eb6834", "#eda100"][j], label=ASSET[sym])
        ax.set_xticks(x)
        ax.set_xticklabels(["low", "mid", "high"])
        ax.set_xlabel("volatility tercile")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("std. of normalized target")
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_lsweep(ls, out):
    Ls = sorted(ls.L.unique())
    if len(Ls) < 2:
        return
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.4), sharey=True)
    for ax, sym in zip(axes, ASSETS):
        for n in ("anchored", "winZ", "winMM", "global"):
            for m in ("lstm", "transformer"):
                g = ls[(ls.symbol == sym) & (ls.model == m) & (ls.norm == n)].groupby("L")["U"].agg(["mean", "std"]).reindex(Ls)
                ax.errorbar(Ls, g["mean"], yerr=g["std"], marker=MARK[m], ms=3, lw=1, capsize=2, color=COLORS[n],
                            ls="-" if m == "lstm" else "--", mfc="white" if m == "transformer" else None,
                            label=f"{NORM[n]}, {MODEL[m]}" if sym == "BTCUSDT" else None)
        ax.axhline(1, color="#444444", lw=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ls)
        ax.set_xticklabels([str(L) for L in Ls])
        ax.set_yscale("log")
        ax.set_title(ASSET[sym], fontsize=9)
        ax.set_xlabel("$L$ (hours)")
    axes[0].set_ylabel("Theil's $U$")
    fig.legend(fontsize=6, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out)
    plt.close(fig)


def manchetes(runs, z, a, t, d, seeds, ro, ls):
    h1 = runs[runs.H == 1]
    print("\n=== MANCHETES")
    deep = h1[h1.model.isin(["lstm", "transformer"])]
    for n in NORMS:
        g = deep[deep.norm == n].groupby(["symbol", "model"])["U"].mean()
        print(f"U deep {n:9s}: {g.min():.4f}-{g.max():.4f}")
        g = h1[(h1.norm == n) & (h1.model == "linear")].groupby("symbol")["U"].mean()
        print(f"U linear {n:9s}: {g.min():.4f}-{g.max():.4f}")
    an = h1[h1.norm == "anchored"]
    print("anchored DM deep: ", an[an.model != "linear"].dm_stat.min().round(2), an[an.model != "linear"].dm_stat.max().round(2),
          "| linear:", an[an.model == "linear"].dm_stat.min().round(2), an[an.model == "linear"].dm_stat.max().round(2))
    print("anchored bet_ratio deep:", an[an.model != "linear"].bet_ratio.min().round(3), an[an.model != "linear"].bet_ratio.max().round(3))
    print("anchored corr_returns:", an.corr_returns.min().round(3), an.corr_returns.max().round(3), "dir_acc", an.dir_acc.min().round(3), an.dir_acc.max().round(3))
    print("global U range all:", h1[h1.norm == "global"].U.min().round(2), h1[h1.norm == "global"].U.max().round(2),
          "sd over seeds max", h1[h1.norm == "global"].groupby(["symbol", "model"])["U"].std().max().round(2))
    print("min U overall (cell mean):", h1.groupby(["symbol", "model", "norm"])["U"].mean().min().round(4))
    print("beats persistence (U<1) cells:", (h1.groupby(["symbol", "model", "norm"])["U"].mean() < 1).sum())
    print("zero-output ratios global/persist:", (z[z.norm == "global"].rmse_zero / z[z.norm == "global"].rmse_persist).round(0).tolist(),
          "winZ:", (z[z.norm == "winZ"].rmse_zero / z[z.norm == "winZ"].rmse_persist).round(1).tolist(),
          "winMM:", (z[z.norm == "winMM"].rmse_zero / z[z.norm == "winMM"].rmse_persist).round(1).tolist())
    al = a[a.model == "lstm"]
    print("cov share (non-global):", (al[al.norm != "global"].cov_term / al[al.norm != "global"].mse).min().round(3),
          (al[al.norm != "global"].cov_term / al[al.norm != "global"].mse).max().round(3))
    print("anchored mean eps2:", al[al.norm == "anchored"].mean_eps2.round(3).tolist(), "winZ mean eps2:", al[al.norm == "winZ"].mean_eps2.round(3).tolist())
    print("dispersion ratio T3/T1 anchored:", d.ratio_anchored.round(2).tolist(), "rellast:", d.ratio_rellast.round(2).tolist())
    print("terciles winZ high:", t[(t.norm == "winZ")].U_T3.min().round(3), t[(t.norm == "winZ")].U_T3.max().round(3),
          "low:", t[(t.norm == "winZ")].U_T1.min().round(3), t[(t.norm == "winZ")].U_T1.max().round(3))
    print("seeds t anchored vs winZ:", seeds[seeds.vs == "winZ"].t.min().round(1), seeds[seeds.vs == "winZ"].t.max().round(1),
          "| vs winMM:", seeds[seeds.vs == "winMM"].t.min().round(1), seeds[seeds.vs == "winMM"].t.max().round(1),
          "| n_neg min", seeds.n_neg.min())
    for H in (6, 24):
        g = runs[(runs.H == H) & (runs.norm == "anchored") & (runs.model != "linear")]
        print(f"H={H} anchored deep U {g.U.min():.4f}-{g.U.max():.4f} DM {g.dm_stat.min():.2f}..{g.dm_stat.max():.2f}; "
              f"global U {runs[(runs.H == H) & (runs.norm == 'global')].U.min():.2f}-{runs[(runs.H == H) & (runs.norm == 'global')].U.max():.2f}")
    if ro is not None and not ro.empty:
        print("rolling: anchored rank counts", ro[ro.norm == "anchored"].groupby("model").size().to_dict())
    print("L sweep levels:", sorted(ls.L.unique()))
    print("durations (min, mean by model):", (runs[runs.H == 1].groupby("model").duration_s.mean() / 60).round(0).to_dict(),
          "epochs:", runs[runs.H == 1].groupby("model").epochs.mean().round(0).to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", required=True)
    ap.add_argument("--out", required=True)
    a_ = ap.parse_args()
    T, F = os.path.join(a_.out, "tables"), os.path.join(a_.out, "figures")
    os.makedirs(T, exist_ok=True); os.makedirs(F, exist_ok=True)
    runs = pd.read_csv(os.path.join(a_.res, "runs.csv"))
    z = pd.read_csv(os.path.join(a_.res, "zero_output.csv"))
    a = pd.read_csv(os.path.join(a_.res, "amplification.csv"))
    t = pd.read_csv(os.path.join(a_.res, "terciles.csv"))
    d = pd.read_csv(os.path.join(a_.res, "dispersion.csv"))
    hs = pd.read_csv(os.path.join(a_.res, "horizon_steps.csv"))
    def _csv(nome, cols):
        try:
            return pd.read_csv(os.path.join(a_.res, nome))
        except pd.errors.EmptyDataError:
            return pd.DataFrame(columns=cols)
    ro = _csv("rolling.csv", ["grid", "symbol", "origin", "end_date", "model", "norm", "rmse", "rmse_persist", "U", "dm_stat"])
    ls = _csv("lsweep.csv", ["L", "symbol", "model", "norm", "seed", "rmse", "rmse_persist", "U"])
    h1 = runs[runs.H == 1]
    tab_zero(z, os.path.join(T, "tab1_zero_output.tex"))
    tab_amp(a, os.path.join(T, "tab2_amplification.tex"))
    tab_main(h1, os.path.join(T, "tab3_main.tex"), os.path.join(T, "tabA1_main_app.tex"))
    tab_horizon(runs, os.path.join(T, "tab4_horizon.tex"))
    tab_terciles(t, os.path.join(T, "tab5_terciles.tex"))
    seeds = tab_seeds(h1, os.path.join(T, "tab6_seeds.tex"))
    seeds.to_csv(os.path.join(a_.res, "seeds_ttest.csv"), index=False)
    ro2 = tab_rolling(ro, os.path.join(T, "tab7_rolling.tex"))
    Ls = tab_lsweep(ls, os.path.join(T, "tab8_lsweep.tex"))
    mean_tr, max_tr = fig_implicit(runs, os.path.join(F, "fig1_implicit_btc.pdf"))
    fig_U(h1, os.path.join(F, "fig2_U_by_norm.pdf"))
    fig_horizon(hs, os.path.join(F, "fig3_horizon_sqrt.pdf"))
    fig_dispersion(d, os.path.join(F, "fig4_dispersion.pdf"))
    fig_lsweep(ls, os.path.join(F, "fig5_lsweep.pdf"))
    print(f"BTC training mean {mean_tr:.0f} max {max_tr:.0f}; rolling table: {'sim' if ro2 is not None else 'nao'}; L sweep: {Ls}")
    manchetes(runs, z, a, t, d, seeds, ro2, ls)


if __name__ == "__main__":
    main()
