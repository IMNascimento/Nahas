#!/usr/bin/env python
"""
Artigo 4 — decomposicao de variancia do log RMSE e ablacoes.

Le as grades H=1 (artigo1_principal_H1, artigo1_sementes_H1*, artigo4_L*_H1*), a
replica H=24 (artigo1_principal_H24 + artigo4_sementes_H24*) e as ablacoes
(artigo4_ablacao_A1/A2/A3), e escreve em --out:

  cells.csv          uma linha por celula (norm, arch, seed, L, asset, H, log_rmse, U, ...)
  anova.csv          ANOVA tipo II do modelo (eq. do artigo): df, SS, F, p parametrico,
                     p por permutacao (rotulos do fator permutados dentro dos estratos
                     dos demais), eta2 parcial, omega2; termos ordenados por omega2
  varcomp.csv        componentes de variancia do modelo misto (semente aleatoria)
  tukey.csv          Tukey HSD entre normalizacoes dentro de cada arquitetura
  ratios.csv         razao de RMSE entre arquiteturas por normalizacao (media e sd sobre sementes)
  anova_H24.csv      a mesma decomposicao na replica H=24 (termos comuns)
  ablations.csv      Delta = log RMSE(global) - log RMSE(janela Z) por semente, com e sem
                     ablacao; t pareado e contagem de sinais (H1, H2, H3)

    ./venv/bin/python analysis/artigo4_anova.py --out results/artigo4 [--perms 1000]
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
from analysis.artigo1_resultados import rotulo_norm  # noqa: E402

GRIDS_H1 = ["artigo1_principal_H1", "artigo1_sementes_H1", "artigo1_sementes_H1_linear", "artigo1_sementes_H1_evomsn"]
GRIDS_H24 = ["artigo1_principal_H24", "artigo4_sementes_H24", "artigo4_sementes_H24_linear", "artigo4_sementes_H24_evomsn"]
ABLATIONS = {"A1": "artigo4_ablacao_A1", "A2": "artigo4_ablacao_A2", "A3": "artigo4_ablacao_A3"}
FACTORS = ["norm", "arch", "L", "asset"]
TERMS = ["norm", "arch", "L", "asset", "norm:arch", "norm:L", "norm:asset", "arch:L", "arch:asset", "L:asset"]


def _ler(grid_root, grids, extra=None):
    rows = []
    for g in grids:
        for mp in glob.glob(os.path.join(grid_root, g, "*", "metrics.json")):
            d = os.path.dirname(mp)
            pp = os.path.join(d, "provenance.json")
            if not os.path.exists(pp):
                continue
            m, p = json.load(open(mp)), json.load(open(pp))
            b, cfg = m["baselines"], p["config"]
            hp = p.get("trainer_hyperparameters", {})
            rows.append({"grid": g, "run_dir": d, "H": int(cfg.get("steps_ahead", 1)), "L": int(cfg.get("window_size", 96)),
                         "asset": cfg["symbol"], "arch": p["model_type"], "norm": rotulo_norm(cfg.get("normalization", {})),
                         "seed": int(p["seed"]), "rmse": b["model"]["rmse"], "rmse_persist": b["persistence"]["rmse"],
                         "log_rmse": float(np.log(b["model"]["rmse"])), "U": b["model"]["rmse"] / b["persistence"]["rmse"],
                         "use_layernorm": hp.get("use_layernorm"), "positional_encoding": hp.get("positional_encoding"),
                         "layer_norm": hp.get("layer_norm"), "duration_s": p.get("duration_s"), "commit": (p.get("code") or {}).get("commit")})
    return pd.DataFrame(rows)


def cells_h1(grid_root):
    grids = GRIDS_H1 + [os.path.basename(g) for g in glob.glob(os.path.join(grid_root, "artigo4_L*_H1*"))]
    df = _ler(grid_root, grids)
    df = df[df.H == 1]
    # uma celula por (norm, arch, seed, L, asset): se houver duplicata (reexecucao), fica a mais recente
    df = df.sort_values("run_dir").drop_duplicates(["norm", "arch", "seed", "L", "asset"], keep="last")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- ANOVA
def _formula(df):
    levels = {f: df[f].nunique() for f in FACTORS}
    main = [f"C({f})" for f in FACTORS if levels[f] > 1]
    inter = [f"C({a}):C({b})" for a, b in [t.split(":") for t in TERMS if ":" in t] if levels[a] > 1 and levels[b] > 1]
    return "log_rmse ~ " + " + ".join(main + inter + ["C(seed)"])


def anova(df, perms=0, rng=None):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    f = _formula(df)
    fit = smf.ols(f, data=df).fit()
    tab = anova_lm(fit, typ=2)
    ss_res, df_res = tab.loc["Residual", "sum_sq"], tab.loc["Residual", "df"]
    ms_res = ss_res / df_res
    ss_tot = tab["sum_sq"].sum()
    out = []
    for term, r in tab.iterrows():
        if term == "Residual":
            continue
        out.append({"term": term.replace("C(", "").replace(")", ""), "df": r["df"], "ss": r["sum_sq"], "F": r["F"], "p": r["PR(>F)"],
                    "eta2p": r["sum_sq"] / (r["sum_sq"] + ss_res), "omega2": (r["sum_sq"] - r["df"] * ms_res) / (ss_tot + ms_res)})
    res = pd.DataFrame(out)
    res["p_perm"] = np.nan
    if perms:
        rng = rng or np.random.default_rng(1482563973)
        for i, row in res.iterrows():
            term = row["term"]
            if term == "seed":
                continue
            facs = term.split(":")
            strata = [c for c in FACTORS + ["seed"] if c not in facs and df[c].nunique() > 1]
            count = 0
            for _ in range(perms):
                d2 = df.copy()
                # permuta os rotulos do(s) fator(es) do termo dentro dos estratos dos demais fatores
                d2[facs] = d2.groupby(strata, group_keys=False)[facs].transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(2**31 - 1))).to_numpy()) if strata else d2[facs].sample(frac=1).to_numpy()
                F = anova_lm(smf.ols(f, data=d2).fit(), typ=2).loc[f"C({term})" if ":" not in term else ":".join(f"C({x})" for x in facs), "F"]
                count += F >= row["F"]
            res.loc[i, "p_perm"] = (count + 1) / (perms + 1)
    res["residual_ss"], res["residual_df"] = ss_res, df_res
    return res.sort_values("omega2", ascending=False).reset_index(drop=True), fit


def varcomp(df):
    import statsmodels.formula.api as smf
    f = _formula(df).replace(" + C(seed)", "")
    md = smf.mixedlm(f, data=df, groups=df["seed"]).fit(reml=True)
    return pd.DataFrame([{"component": "seed", "variance": float(md.cov_re.iloc[0, 0])}, {"component": "residual", "variance": float(md.scale)}]), md


def tukey(df):
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    out = []
    for arch, g in df.groupby("arch"):
        cm = g.groupby(["norm", "seed", "L", "asset"])["log_rmse"].mean().reset_index()
        r = pairwise_tukeyhsd(cm["log_rmse"], cm["norm"], alpha=0.05)
        t = pd.DataFrame(r.summary().data[1:], columns=r.summary().data[0])
        t.insert(0, "arch", arch)
        out.append(t)
    return pd.concat(out, ignore_index=True)


def ratios(df):
    rows = []
    for (norm, L, asset, seed), g in df.groupby(["norm", "L", "asset", "seed"]):
        r = g.set_index("arch")["rmse"]
        for a, b in [("lstm", "transformer"), ("lstm", "linear"), ("transformer", "linear")]:
            if a in r and b in r:
                rows.append({"norm": norm, "L": L, "asset": asset, "seed": seed, "pair": f"{a}/{b}", "ratio": r[a] / r[b]})
    d = pd.DataFrame(rows)
    return d.groupby(["norm", "L", "asset", "pair"])["ratio"].agg(["mean", "std", "count"]).reset_index()


def ablations(grid_root, base):
    """Delta = log RMSE(global) - log RMSE(janela Z) por semente; referencia = base (L=96, H=1, BTC/ETH)."""
    from scipy import stats
    ab = _ler(grid_root, list(ABLATIONS.values()))
    rows = []
    if ab.empty:
        return pd.DataFrame(rows)
    ref = base[(base.L == 96) & base.asset.isin(["BTCUSDT", "ETHUSDT"])]
    for key, g in ABLATIONS.items():
        d = ab[ab.grid == g]
        if d.empty:
            continue
        arch = "transformer" if key in ("A1", "A2") else "lstm"
        for asset in ("BTCUSDT", "ETHUSDT"):
            def delta(src):
                s = src[(src.asset == asset) & (src.arch == arch)].pivot_table(index="seed", columns="norm", values="log_rmse")
                return (s["global"] - s["winZ"]).dropna(), s
            d_ref, s_ref = delta(ref)
            d_abl, s_abl = delta(d)
            seeds = sorted(set(d_ref.index) & set(d_abl.index))
            if not seeds:
                continue
            x, y = d_abl.loc[seeds].to_numpy(), d_ref.loc[seeds].to_numpy()
            tt = stats.ttest_rel(x, y) if len(seeds) > 1 else None
            rows.append({"ablation": key, "arch": arch, "asset": asset, "n_seeds": len(seeds),
                         "delta_ref_mean": y.mean(), "delta_abl_mean": x.mean(), "diff_mean": (x - y).mean(),
                         "t": tt.statistic if tt else np.nan, "p": tt.pvalue if tt else np.nan,
                         "n_sign_up": int(((x - y) > 0).sum()), "n_sign_down": int(((x - y) < 0).sum()),
                         "delta_ref_per_seed": ";".join(f"{v:.3f}" for v in y), "delta_abl_per_seed": ";".join(f"{v:.3f}" for v in x),
                         "level_winZ_ref": s_ref["winZ"].loc[seeds].mean(), "level_winZ_abl": s_abl["winZ"].loc[seeds].mean(),
                         "level_anchored_ref": s_ref.get("anchored", pd.Series(dtype=float)).reindex(seeds).mean(),
                         "level_anchored_abl": s_abl.get("anchored", pd.Series(dtype=float)).reindex(seeds).mean()})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "grid"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = cells_h1(a.grid_root)
    df.to_csv(os.path.join(a.out, "cells.csv"), index=False)
    print(f"celulas H=1: {len(df)}  L={sorted(df.L.unique())}  sementes={df.seed.nunique()}  normalizacoes={sorted(df.norm.unique())}")
    an, fit = anova(df, perms=a.perms)
    an.to_csv(os.path.join(a.out, "anova.csv"), index=False)
    print(an[["term", "df", "F", "p", "p_perm", "eta2p", "omega2"]].round(4).to_string())
    vc, md = varcomp(df)
    vc.to_csv(os.path.join(a.out, "varcomp.csv"), index=False); print(vc.round(5).to_string())
    tukey(df).to_csv(os.path.join(a.out, "tukey.csv"), index=False)
    ratios(df).to_csv(os.path.join(a.out, "ratios.csv"), index=False)
    h24 = _ler(a.grid_root, GRIDS_H24)
    h24 = h24[h24.H == 24].drop_duplicates(["norm", "arch", "seed", "L", "asset"], keep="last")
    if len(h24) and h24.seed.nunique() > 1:
        an24, _ = anova(h24, perms=a.perms)
        an24.to_csv(os.path.join(a.out, "anova_H24.csv"), index=False); print("H24 celulas", len(h24))
    else:
        print(f"H24: {len(h24)} celulas, {h24.seed.nunique() if len(h24) else 0} semente(s); replica ainda incompleta")
    ab = ablations(a.grid_root, df)
    ab.to_csv(os.path.join(a.out, "ablations.csv"), index=False)
    print("ablacoes:", len(ab), "linhas")
    if len(ab):
        print(ab[["ablation", "asset", "delta_ref_mean", "delta_abl_mean", "diff_mean", "t", "n_sign_up", "n_sign_down"]].round(3).to_string())


if __name__ == "__main__":
    main()
