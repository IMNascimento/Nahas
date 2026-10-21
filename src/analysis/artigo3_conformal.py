#!/usr/bin/env python
"""
Artigo 3 — intervalos conformais com escala local, calculados a posteriori sobre
as predicoes exportadas pela grade do artigo 1 (nenhum treino novo).

Para cada horizonte H in {1, 6, 24}, ativo, backbone e normalizacao (global
Z-score, Z-score por janela, ancorada), quatro construcoes:
  split     — split conformal com residuo bruto;
  lsc       — split conformal com score |y − ŷ| / s(W)  (proposta);
  aci_raw   — ACI (gamma = 0,005) sobre o residuo bruto;
  aci_lsc   — ACI sobre o score escalado.
E a persistencia (ŷ = p_t) com a escala ancorada como s.

Calibracao = particao de validacao; avaliacao = particao de teste. Metricas:
cobertura, largura media em fracao do preco de origem p_t, interval score
relativo (Gneiting & Raftery, em fracao de p_t), cobertura e largura por tercil
de volatilidade realizada causal (desvio dos 24 log-retornos que terminam na
origem t), gap = cob(tercil baixo) − cob(tercil alto), e, no ACI, numero de
horas com alpha_t fora de (0, 1).

ACI e calculado so no ultimo passo h = H (custo); split/lsc em todos os passos.

    python analysis/artigo3_conformal.py --out results/artigo3
    python analysis/artigo3_conformal.py --out results/artigo3 --only H=1,symbol=BTCUSDT  (depuracao)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.conformal import adaptive_conformal, interval_score, split_conformal   # noqa: E402

GRIDS = {1: "artigo1_principal_H1", 6: "artigo1_principal_H6", 24: "artigo1_principal_H24"}
ALPHAS = (0.10, 0.05)
GAMMA = 0.005
VOL_WINDOW = 24


def rotulo_norm(n: dict):
    if n.get("strategy") == "global" and n.get("scaler_type") == "standard":
        return "global"
    if n.get("strategy") == "local" and n.get("y_mode") == "zscore_target":
        return "window"
    if n.get("strategy") == "local" and n.get("y_mode") == "anchored":
        return "anchored"
    return None


def listar_execucoes(grid_root: str):
    """[(symbol, model, norm, pasta)] das execucoes com CSV de teste e validacao."""
    out = []
    for p in sorted(glob.glob(os.path.join(grid_root, "*", "provenance.json"))):
        j = json.load(open(p))
        norm = rotulo_norm(j["config"].get("normalization", {}))
        if norm is None:
            continue
        d = os.path.dirname(p)
        te = [c for c in glob.glob(os.path.join(d, "csv", "results_*.csv")) if "results_val_" not in c]
        va = glob.glob(os.path.join(d, "csv", "results_val_*.csv"))
        if te and va:
            out.append((j["config"]["symbol"], j["model_type"], norm, d, te[0], va[0]))
    return out


_PRECOS = {}


def precos(symbol: str) -> pd.Series:
    """Fechamentos horarios do banco local, indexados por timestamp."""
    if symbol not in _PRECOS:
        from database.model_nocapital import PriceHistory
        df = PriceHistory.get_between_dates(start_date="2022-06-01 00:00:00", end_date="2024-11-12 00:00:00",
                                            symbol=symbol, interval="1h")
        s = pd.Series(df["close"].to_numpy(float), index=pd.to_datetime(df["timestamp"]))
        _PRECOS[symbol] = s[~s.index.duplicated(keep="last")].sort_index()
    return _PRECOS[symbol]


def volatilidade_causal(px: pd.Series, window: int = VOL_WINDOW) -> pd.Series:
    """Desvio dos `window` log-retornos que terminam em t (inclusive), indexado por t."""
    lr = np.log(px).diff()
    return lr.rolling(window).std()


def alinhar(df: pd.DataFrame, H: int, px: pd.Series) -> pd.DataFrame:
    """
    Convencao do CSVExporter: com H = 1 o timestamp e o do alvo; com H > 1 o
    timestamp e o do ultimo passo (t + H). Em ambos a origem e t = ts − H horas e
    o alvo do passo h esta em t + h. Confere os valores reais contra o banco.
    """
    ts = pd.to_datetime(df["timestamp"])
    origem = ts - pd.to_timedelta(H, "h")
    alvo = origem + pd.to_timedelta(df["step"].to_numpy(), "h")
    real_db = px.reindex(alvo).to_numpy()
    ok = np.isfinite(real_db)
    bate = ok & np.isclose(df["real_value"].to_numpy(), np.where(ok, real_db, 0.0), rtol=1e-6)
    # Horas ausentes na Binance (ex.: 24/03/2023) deslocam meia duzia de linhas; tolera 1%.
    if bate.mean() < 0.99:
        raise RuntimeError(f"so {bate.mean():.1%} dos valores reais do CSV batem com o banco: convencao de timestamp errada")
    out = df.copy()
    out["origem"] = origem.to_numpy()
    out["p_t"] = px.reindex(origem).to_numpy()
    vol = volatilidade_causal(px)
    out["vol"] = vol.reindex(origem).to_numpy()
    return out


def tercis(v: np.ndarray) -> np.ndarray:
    q1, q2 = np.nanquantile(v, [1 / 3, 2 / 3])
    return np.digitize(v, [q1, q2])


def por_tercil(y, lo, hi, terc, p):
    out = {}
    for k in (0, 1, 2):
        m = terc == k
        w = (hi[m] - lo[m]) / p[m]
        out[f"cov_T{k + 1}"] = float(np.mean((y[m] >= lo[m]) & (y[m] <= hi[m])))
        out[f"width_T{k + 1}"] = float(np.mean(w[np.isfinite(w)])) if np.isfinite(w).any() else np.inf
    out["gap"] = out["cov_T1"] - out["cov_T3"]
    return out


def avaliar(cal: pd.DataFrame, te: pd.DataFrame, alpha: float, com_aci: bool) -> list:
    """Quatro construcoes para um passo; `cal`/`te` ja filtrados no passo."""
    yc, yhc, sc = cal["real_value"].to_numpy(float), cal["predicted_value"].to_numpy(float), cal["scale"].to_numpy(float)
    yt, yht, st = te["real_value"].to_numpy(float), te["predicted_value"].to_numpy(float), te["scale"].to_numpy(float)
    p = te["p_t"].to_numpy(float)
    terc = tercis(te["vol"].to_numpy(float))
    linhas = []

    def registra(nome, r):
        # No ACI, alpha_t <= 0 da quantil infinito e intervalo infinito naquela hora.
        # Largura e interval score (relativos a p_t) sao calculados nas horas finitas,
        # e o numero de horas infinitas e reportado ao lado.
        lo, hi = r["lo"], r["hi"]
        w = (hi - lo) / p
        fin = np.isfinite(w)
        with np.errstate(invalid="ignore"):
            is_rel = interval_score(yt[fin] / p[fin], lo[fin] / p[fin], hi[fin] / p[fin], alpha) if fin.any() else np.nan
        linhas.append({"construction": nome, "alpha": alpha, "coverage": r["coverage"],
                       "rel_width": float(np.mean(w[fin])) if fin.any() else np.inf,
                       "median_rel_width": float(np.median(w[fin])) if fin.any() else np.inf,
                       "n_inf": int((~fin).sum()), "is_rel": is_rel,
                       "n_alpha_out": r.get("n_alpha_out_of_range"), "n": r["n"],
                       **por_tercil(yt, lo, hi, terc, p)})
        return r

    registra("split", split_conformal(yc, yhc, yt, yht, alpha, level_test=p))
    registra("lsc", split_conformal(yc, yhc, yt, yht, alpha, scale_cal=sc, scale_test=st, level_test=p))
    if com_aci:
        r1 = registra("aci_raw", adaptive_conformal(yc, yhc, yt, yht, alpha, GAMMA, level_test=p))
        r2 = registra("aci_lsc", adaptive_conformal(yc, yhc, yt, yht, alpha, GAMMA, scale_cal=sc, scale_test=st, level_test=p))
        return linhas, {"aci_raw": r1["alpha_path"], "aci_lsc": r2["alpha_path"], "timestamp": te["timestamp"].to_numpy()}
    return linhas, None


def persistencia(df: pd.DataFrame) -> pd.DataFrame:
    """ŷ = p_t com a escala ancorada do mesmo ativo/horizonte (depende so da janela)."""
    out = df.copy()
    out["predicted_value"] = out["p_t"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "grid"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="filtros: H=1,symbol=BTCUSDT,model=lstm")
    ap.add_argument("--sem-aci", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    filtros = dict(kv.split("=") for kv in a.only.split(",") if kv)

    linhas, paths = [], {}
    t0 = time.time()
    for H, grid in GRIDS.items():
        if "H" in filtros and int(filtros["H"]) != H:
            continue
        execs = listar_execucoes(os.path.join(a.grid_root, grid))
        persist_feita = set()
        for symbol, model, norm, d, te_csv, va_csv in execs:
            if filtros.get("symbol", symbol) != symbol or filtros.get("model", model) != model:
                continue
            px = precos(symbol)
            te = alinhar(pd.read_csv(te_csv), H, px)
            va = alinhar(pd.read_csv(va_csv), H, px)
            te = te[np.isfinite(te["p_t"]) & np.isfinite(te["vol"])]
            va = va[np.isfinite(va["p_t"])]
            jogos = [(symbol, model, norm, va, te)]
            if norm == "anchored" and symbol not in persist_feita:
                jogos.append((symbol, "persistence", "anchored", persistencia(va), persistencia(te)))
                persist_feita.add(symbol)
            for sym, mod, nrm, cal_df, te_df in jogos:
                for h in range(1, H + 1):
                    cal_h, te_h = cal_df[cal_df["step"] == h], te_df[te_df["step"] == h]
                    for alpha in ALPHAS:
                        com_aci = (h == H) and not a.sem_aci
                        ls, ap_ = avaliar(cal_h, te_h, alpha, com_aci)
                        for l in ls:
                            linhas.append({"H": H, "step": h, "symbol": sym, "model": mod, "norm": nrm,
                                           "run_dir": os.path.basename(d) if mod != "persistence" else "", **l})
                        if ap_ is not None:
                            paths[f"H{H}_{sym}_{mod}_{nrm}_a{alpha}"] = ap_
                print(f"[{time.time() - t0:6.0f}s] H={H:2d} {sym:8s} {mod:12s} {nrm:9s} ok", flush=True)

    df = pd.DataFrame(linhas)
    df.to_csv(os.path.join(a.out, "resultados.csv"), index=False)
    # trajetorias de alpha_t do ACI (figura 3)
    np.savez_compressed(os.path.join(a.out, "aci_alpha_paths.npz"),
                        **{k + "__" + c: v for k, p in paths.items() for c, v in p.items()})
    print(f"\n{len(df)} linhas em {a.out}/resultados.csv ({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
