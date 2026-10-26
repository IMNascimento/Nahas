#!/usr/bin/env python
"""
Artigo 1 — consolida as execucoes das grades (metrics.json + provenance.json + CSVs
de teste) e os calculos a posteriori que o texto pede, sem treinar nada:

  runs.csv          uma linha por execucao (H=1 com 5 sementes; H=6 e H=24 com 1):
                    U = RMSE/RMSE_persistencia, DM, correlacao de retornos, acerto
                    direcional, razao de aposta, semente, grade, caminhos.
  zero_output.csv   previsao implicita (saida nula) por normalizacao x ativo em H=1:
                    ancorada -> p_t; janela Z -> media da janela; janela MinMax -> minimo
                    da janela; global Z -> media do treino. Calculada a partir dos precos
                    do banco, nas mesmas origens do CSV de teste (semente principal).
  amplification.csv Proposicao 2: media de s^2, media de eps^2, produto, MSE realizado e
                    termo de covariancia, por normalizacao x ativo x backbone (H=1,
                    semente principal), com s lido da coluna `scale` do CSV.
  terciles.csv      U por tercil de volatilidade causal (desvio dos 24 log-retornos que
                    terminam em t-1) para as configuracoes principais, e dispersao do
                    alvo normalizado por tercil sob a ancorada e sob relative-to-last.
  horizon_steps.csv RMSE por passo h das execucoes H=24 (ancorada), para comparar com
                    sqrt(h) x RMSE(1).
  rolling.csv       (se houver results/grid/artigo1_rolling_*) U por origem e configuracao.
  lsweep.csv        (se houver results/grid/artigo4_L*_H1) U por L para ancorada e janela Z.

    ./venv/bin/python analysis/artigo1_resultados.py --out results/artigo1
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

GRIDS_H1 = ["artigo1_principal_H1", "artigo1_sementes_H1", "artigo1_sementes_H1_linear", "artigo1_sementes_H1_evomsn"]
GRIDS_H = {6: ["artigo1_principal_H6"], 24: ["artigo1_principal_H24"]}
SEED_PRINCIPAL = 1482563973
ASSETS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
NORMS = ["global", "winZ", "winMM", "anchored", "evomsn"]
VOL_WINDOW = 24


def rotulo_norm(n: dict) -> str:
    s, y = n.get("strategy"), n.get("y_mode")
    if s == "global":
        return "global"
    if s == "evomsn":
        return "evomsn"
    return {"zscore_target": "winZ", "minmax_target": "winMM", "anchored": "anchored", "relative_last": "relLast"}.get(y, y or "?")


def _csvs(d):
    te = [c for c in glob.glob(os.path.join(d, "csv", "results_*.csv")) if "results_val_" not in c]
    va = glob.glob(os.path.join(d, "csv", "results_val_*.csv"))
    return (te[0] if te else None), (va[0] if va else None)


def carregar(grid_root: str) -> pd.DataFrame:
    linhas = []
    for H, grids in [(1, GRIDS_H1), *GRIDS_H.items()]:
        for g in grids:
            for mp in sorted(glob.glob(os.path.join(grid_root, g, "*", "metrics.json"))):
                d = os.path.dirname(mp)
                pp = os.path.join(d, "provenance.json")
                if not os.path.exists(pp):
                    continue
                m, p = json.load(open(mp)), json.load(open(pp))
                b = m.get("baselines", {})
                cfg = p["config"]
                te, va = _csvs(d)
                linhas.append({
                    "grid": g, "run_dir": d, "H": H, "L": int(cfg.get("window_size", 96)),
                    "symbol": cfg["symbol"], "model": p["model_type"], "norm": rotulo_norm(cfg.get("normalization", {})),
                    "seed": int(p["seed"]), "rmse": b["model"]["rmse"], "mae": b["model"]["mae"], "r2": b["model"]["r2"],
                    "rmse_persist": b["persistence"]["rmse"], "mae_persist": b["persistence"]["mae"],
                    "U": b["model"]["rmse"] / b["persistence"]["rmse"],
                    "dm_stat": b["dm_model_vs_persistence"]["dm_stat"], "dm_p": b["dm_model_vs_persistence"]["p_value"],
                    "corr_returns": b["mimicry"]["corr_returns"], "dir_acc": b["mimicry"]["directional_accuracy"],
                    "bet_ratio": b["mimicry"]["bet_size_ratio"], "n_test": b["mimicry"]["n"],
                    "epochs": p.get("epochs_trained"), "duration_s": p.get("duration_s"),
                    "train_first": p["split"]["train"]["first_timestamp"], "train_last": p["split"]["train"]["last_timestamp"],
                    "test_csv": te, "val_csv": va, "commit": (p.get("code") or {}).get("commit"),
                })
    df = pd.DataFrame(linhas)
    return df


# ---------------------------------------------------------------- precos e alinhamento
_PRECOS = {}


def precos(symbol: str) -> pd.Series:
    if symbol not in _PRECOS:
        from database.model_nocapital import PriceHistory
        df = PriceHistory.get_between_dates(start_date="2017-08-01 00:00:00", end_date="2024-11-12 00:00:00",
                                            symbol=symbol, interval="1h")
        s = pd.Series(df["close"].to_numpy(float), index=pd.to_datetime(df["timestamp"]))
        _PRECOS[symbol] = s[~s.index.duplicated(keep="last")].sort_index()
    return _PRECOS[symbol]


def alinhar(df: pd.DataFrame, H: int, px: pd.Series, L: int = 96) -> pd.DataFrame:
    """Origem t = timestamp - H h (convencao do CSVExporter); confere o real contra o banco."""
    ts = pd.to_datetime(df["timestamp"])
    origem = ts - pd.to_timedelta(H, "h")
    alvo = origem + pd.to_timedelta(df["step"].to_numpy(), "h")
    real_db = px.reindex(alvo).to_numpy()
    ok = np.isfinite(real_db)
    bate = ok & np.isclose(df["real_value"].to_numpy(), np.where(ok, real_db, 0.0), rtol=1e-6)
    if bate.mean() < 0.99:
        raise RuntimeError(f"so {bate.mean():.1%} dos valores reais batem com o banco")
    out = df.copy()
    out["origem"] = origem.to_numpy()
    out["p_t"] = px.reindex(origem).to_numpy()
    lr = np.log(px).diff()
    vol = lr.rolling(VOL_WINDOW).std().shift(1)            # 24 retornos que terminam em t-1
    out["vol"] = vol.reindex(origem).to_numpy()
    # estatisticas da janela de L fechamentos que termina em t (para a saida nula de janela Z / MinMax)
    out["win_mean"] = px.rolling(L).mean().reindex(origem).to_numpy()
    out["win_min"] = px.rolling(L).min().reindex(origem).to_numpy()
    return out


def rmse(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)))


def tercis(v):
    q1, q2 = np.nanquantile(v, [1 / 3, 2 / 3])
    return np.digitize(v, [q1, q2])


# ---------------------------------------------------------------- calculos a posteriori (H=1, semente principal)
def zero_output(df: pd.DataFrame) -> pd.DataFrame:
    """RMSE da previsao implicita por normalizacao x ativo (LSTM da semente principal como referencia de linhas)."""
    out = []
    base = df[(df.H == 1) & (df.seed == SEED_PRINCIPAL) & (df.model == "lstm")]
    for sym in ASSETS:
        px = precos(sym)
        for norm in NORMS:
            r = base[(base.symbol == sym) & (base.norm == norm)]
            if r.empty:
                continue
            r = r.iloc[0]
            te = alinhar(pd.read_csv(r.test_csv), 1, px)
            y, p = te["real_value"].to_numpy(float), te["p_t"].to_numpy(float)
            if norm == "anchored":
                z = p
            elif norm == "winZ":
                z = te["win_mean"].to_numpy(float)
            elif norm == "winMM":
                z = te["win_min"].to_numpy(float)
            elif norm == "global":
                tr = px[(px.index >= pd.Timestamp(r.train_first)) & (px.index <= pd.Timestamp(r.train_last))]
                z = np.full_like(y, tr.mean())
            else:
                z = None
            # redes treinadas: media sobre as 5 sementes por backbone
            redes = df[(df.H == 1) & (df.symbol == sym) & (df.norm == norm)].groupby("model")["rmse"].mean()
            out.append({"symbol": sym, "norm": norm, "rmse_zero": rmse(y, z) if z is not None else np.nan,
                        "rmse_persist": rmse(y, p), "equals_persistence": bool(z is not None and np.allclose(z, p)),
                        "train_mean": float(z[0]) if norm == "global" else np.nan,
                        "rmse_lstm": redes.get("lstm"), "rmse_transformer": redes.get("transformer"), "rmse_linear": redes.get("linear"),
                        "n": int(len(y))})
    return pd.DataFrame(out)


def amplification(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    base = df[(df.H == 1) & (df.seed == SEED_PRINCIPAL)]
    for _, r in base.iterrows():
        te = pd.read_csv(r.test_csv)
        y, yh, s = te["real_value"].to_numpy(float), te["predicted_value"].to_numpy(float), te["scale"].to_numpy(float)
        e = y - yh
        eps = e / s
        s2, eps2 = s ** 2, eps ** 2
        mse = float(np.mean(e ** 2))
        out.append({"symbol": r.symbol, "model": r.model, "norm": r.norm, "mean_s2": float(np.mean(s2)),
                    "mean_eps2": float(np.mean(eps2)), "product": float(np.mean(s2) * np.mean(eps2)), "mse": mse,
                    "cov_term": float(mse - np.mean(s2) * np.mean(eps2)),
                    "corr_s2_eps2": float(np.corrcoef(s2, eps2)[0, 1]), "rmse": float(np.sqrt(mse)),
                    "mean_s": float(np.mean(s)), "n": int(len(y))})
    return pd.DataFrame(out)


def terciles(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """U por tercil (configuracoes principais, semente principal) e dispersao do alvo normalizado por tercil."""
    u_rows, disp_rows = [], []
    base = df[(df.H == 1) & (df.seed == SEED_PRINCIPAL) & (df.norm.isin(["anchored", "winZ", "winMM", "global"]))]
    for sym in ASSETS:
        px = precos(sym)
        for _, r in base[base.symbol == sym].iterrows():
            te = alinhar(pd.read_csv(r.test_csv), 1, px)
            te = te[np.isfinite(te["vol"])]
            terc = tercis(te["vol"].to_numpy(float))
            y, yh, p = te["real_value"].to_numpy(float), te["predicted_value"].to_numpy(float), te["p_t"].to_numpy(float)
            row = {"symbol": sym, "model": r.model, "norm": r.norm}
            for k in (0, 1, 2):
                m = terc == k
                row[f"U_T{k + 1}"] = rmse(y[m], yh[m]) / rmse(y[m], p[m])
                row[f"rmse_T{k + 1}"] = rmse(y[m], yh[m])
                row[f"rmse_persist_T{k + 1}"] = rmse(y[m], p[m])
            u_rows.append(row)
            if r.norm == "anchored" and r.model == "lstm":
                s = te["scale"].to_numpy(float)
                ya, yr = (y - p) / s, (y - p) / p
                drow = {"symbol": sym}
                for k in (0, 1, 2):
                    m = terc == k
                    drow[f"sd_anchored_T{k + 1}"] = float(np.std(ya[m]))
                    drow[f"sd_rellast_T{k + 1}"] = float(np.std(yr[m]))
                    drow[f"vol_T{k + 1}"] = float(np.mean(te["vol"].to_numpy(float)[m]))
                drow["ratio_anchored"] = drow["sd_anchored_T3"] / drow["sd_anchored_T1"]
                drow["ratio_rellast"] = drow["sd_rellast_T3"] / drow["sd_rellast_T1"]
                disp_rows.append(drow)
    return pd.DataFrame(u_rows), pd.DataFrame(disp_rows)


def horizon_steps(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, r in df[(df.H == 24) & (df.norm.isin(["anchored", "winZ", "global"]))].iterrows():
        te = pd.read_csv(r.test_csv)
        px = precos(r.symbol)
        te = alinhar(te, 24, px)
        for h, g in te.groupby("step"):
            out.append({"symbol": r.symbol, "model": r.model, "norm": r.norm, "h": int(h),
                        "rmse": rmse(g["real_value"], g["predicted_value"]),
                        "rmse_persist": rmse(g["real_value"], g["p_t"])})
    return pd.DataFrame(out)


def rolling(grid_root: str) -> pd.DataFrame:
    out = []
    for gp in sorted(glob.glob(os.path.join(grid_root, "artigo1_rolling_*_o*"))):
        for mp in glob.glob(os.path.join(gp, "*", "metrics.json")):
            d = os.path.dirname(mp)
            pp = os.path.join(d, "provenance.json")
            if not os.path.exists(pp):
                continue
            m, p = json.load(open(mp)), json.load(open(pp))
            b, cfg = m["baselines"], p["config"]
            out.append({"grid": os.path.basename(gp), "symbol": cfg["symbol"], "origin": int(os.path.basename(gp)[-1]),
                        "end_date": cfg["end_date"], "model": p["model_type"], "norm": rotulo_norm(cfg["normalization"]),
                        "rmse": b["model"]["rmse"], "rmse_persist": b["persistence"]["rmse"],
                        "U": b["model"]["rmse"] / b["persistence"]["rmse"], "dm_stat": b["dm_model_vs_persistence"]["dm_stat"],
                        "test_first": p["split"]["test"]["first_timestamp"], "test_last": p["split"]["test"]["last_timestamp"],
                        "n_test": p["split"]["test"]["n"]})
    return pd.DataFrame(out)


def lsweep(grid_root: str, df_h1: pd.DataFrame) -> pd.DataFrame:
    out = []
    for gp in sorted(glob.glob(os.path.join(grid_root, "artigo4_L*_H1"))):
        for mp in glob.glob(os.path.join(gp, "*", "metrics.json")):
            d = os.path.dirname(mp)
            pp = os.path.join(d, "provenance.json")
            if not os.path.exists(pp):
                continue
            m, p = json.load(open(mp)), json.load(open(pp))
            b, cfg = m["baselines"], p["config"]
            out.append({"L": int(cfg["window_size"]), "symbol": cfg["symbol"], "model": p["model_type"],
                        "norm": rotulo_norm(cfg["normalization"]), "seed": int(p["seed"]),
                        "rmse": b["model"]["rmse"], "rmse_persist": b["persistence"]["rmse"],
                        "U": b["model"]["rmse"] / b["persistence"]["rmse"]})
    df = pd.DataFrame(out)
    base = df_h1[df_h1.model.isin(["lstm", "transformer"]) & df_h1.norm.isin(["anchored", "winZ", "winMM", "global"])]
    base = base[["L", "symbol", "model", "norm", "seed", "rmse", "rmse_persist", "U"]]
    return pd.concat([df, base], ignore_index=True) if len(df) else base


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "grid"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = carregar(a.grid_root)
    df.to_csv(os.path.join(a.out, "runs.csv"), index=False)
    print(f"runs: {len(df)}  (H=1: {(df.H == 1).sum()}, H=6: {(df.H == 6).sum()}, H=24: {(df.H == 24).sum()})")
    print(df[df.H == 1].groupby(["symbol", "model", "norm"]).size().describe()[["min", "max"]].to_dict(), "sementes por celula")
    zero_output(df).to_csv(os.path.join(a.out, "zero_output.csv"), index=False); print("zero_output ok")
    amplification(df).to_csv(os.path.join(a.out, "amplification.csv"), index=False); print("amplification ok")
    u, disp = terciles(df)
    u.to_csv(os.path.join(a.out, "terciles.csv"), index=False)
    disp.to_csv(os.path.join(a.out, "dispersion.csv"), index=False); print("terciles ok")
    horizon_steps(df).to_csv(os.path.join(a.out, "horizon_steps.csv"), index=False); print("horizon ok")
    ro = rolling(a.grid_root)
    ro.to_csv(os.path.join(a.out, "rolling.csv"), index=False); print(f"rolling: {len(ro)} execucoes")
    ls = lsweep(a.grid_root, df[df.H == 1])
    ls.to_csv(os.path.join(a.out, "lsweep.csv"), index=False); print(f"lsweep: {len(ls)} execucoes ({sorted(ls.L.unique())})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
