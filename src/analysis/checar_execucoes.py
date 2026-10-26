#!/usr/bin/env python
"""
Checagem continua das execucoes concluidas (grades em results/grid/<grid>/<run>/).
Roda no servidor de tempo em tempo (cron e ao fim de cada item da fila) para pegar
erro sistematico CEDO, e nao depois de dias de GPU. Por execucao nova:

  provenance   existe; seed; versao do keras (LSTM/CNN-LSTM com keras < 3.9 = predicao
               com dropout na inferencia -> marcada PENDENTE de reexecucao)
  metricas     RMSE e persistencia finitos e > 0; U = RMSE/persistencia em [0.5, 50];
               DM finito; n de teste >= 1000
  saida nula   normalizacao ancorada/relative_last -> constant_zero equals_persistence
  csv          sem NaN na predicao; predicao nao constante
  determinismo carrega models/*.keras na CPU e prediz duas vezes a mesma entrada
               aleatoria: diferenca tem de ser 0 (foi exatamente isso que o bug do
               Keras 3.7.0 violava)

Estado em results/checagem_estado.json (so execucoes novas sao checadas; --all refaz
tudo). Saida resumida no stdout e, com --log, anexada a logs/checagem.log.
Codigo de saida 1 se houver FAIL novo.

    CUDA_VISIBLE_DEVICES= python analysis/checar_execucoes.py --log
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERAS_OK = (3, 9)


def _ver(s):
    try:
        return tuple(int(x) for x in str(s).split(".")[:2])
    except Exception:
        return (0, 0)


def checar(run_dir: str, com_modelo: bool = True) -> dict:
    fails, warns, info = [], [], {}
    mp, pp = os.path.join(run_dir, "metrics.json"), os.path.join(run_dir, "provenance.json")
    if not os.path.exists(pp):
        return {"status": "FAIL", "fails": ["sem provenance.json"], "warns": [], "info": info}
    p = json.load(open(pp))
    m = json.load(open(mp))
    cfg = p.get("config", {})
    mt = p.get("model_type")
    info.update(model=mt, symbol=cfg.get("symbol"), H=cfg.get("steps_ahead"), seed=p.get("seed"),
                norm=(cfg.get("normalization") or {}).get("y_mode") or (cfg.get("normalization") or {}).get("strategy"),
                keras=(p.get("environment") or {}).get("packages", {}).get("keras"))
    if p.get("seed") is None:
        fails.append("sem seed")
    if mt in ("lstm", "cnn_lstm") and _ver(info["keras"]) < KERAS_OK:
        return {"status": "PENDENTE", "fails": [], "warns": [f"{mt} treinada com keras {info['keras']}: predicao com dropout; reexecutar"], "info": info}
    b = m.get("baselines", {})
    try:
        rmse, rp = float(b["model"]["rmse"]), float(b["persistence"]["rmse"])
        info["U"] = rmse / rp
        if not (np.isfinite(rmse) and rmse > 0 and np.isfinite(rp) and rp > 0):
            fails.append(f"rmse/persistencia invalidos: {rmse} {rp}")
        elif not (0.5 <= rmse / rp <= 50):
            fails.append(f"U fora da faixa: {rmse / rp:.3f}")
        dm = b.get("dm_model_vs_persistence", {}).get("dm_stat")
        if dm is None or not np.isfinite(dm):
            fails.append("DM ausente ou nao finito")
        n = b.get("mimicry", {}).get("n", 0)
        if n < 1000:
            fails.append(f"n de teste pequeno: {n}")
        info["n"] = n
    except KeyError as e:
        fails.append(f"baselines incompletos: {e}")
    y_mode = (cfg.get("normalization") or {}).get("y_mode")
    cz = b.get("constant_zero_normalized")
    if y_mode in ("anchored", "relative_last"):
        if cz is None:
            warns.append("constant_zero ausente")
        elif not cz.get("equals_persistence"):
            fails.append("ancorada com saida nula != persistencia")
    csvs = [c for c in glob.glob(os.path.join(run_dir, "csv", "results_*.csv")) if "results_val_" not in c]
    if csvs:
        df = pd.read_csv(csvs[0])
        if df["predicted_value"].isna().any():
            fails.append("NaN na predicao")
        if df["predicted_value"].std() == 0:
            fails.append("predicao constante")
        if "scale" in df and (df["scale"] <= 0).any():
            fails.append("scale <= 0 no csv")
    else:
        warns.append("sem csv de teste")
    if com_modelo:
        models = glob.glob(os.path.join(run_dir, "models", "*.keras"))
        if not models:
            warns.append("sem modelo .keras (determinismo nao testado)")
        else:
            try:
                import keras
                model = keras.saving.load_model(models[0], compile=False, safe_mode=False)
                shp = model.input_shape
                shp = shp[0] if isinstance(shp, list) else shp
                x = np.random.default_rng(0).standard_normal((8, *shp[1:])).astype("float32")
                d = float(np.abs(model.predict(x, verbose=0) - model.predict(x, verbose=0)).max())
                info["predict_diff"] = d
                if d != 0.0:
                    fails.append(f"predicao NAO deterministica (max|diff|={d:.3g}): dropout/ruido na inferencia")
                keras.backend.clear_session()
            except Exception as e:
                warns.append(f"nao carregou o modelo: {str(e)[:80]}")
    return {"status": "FAIL" if fails else "OK", "fails": fails, "warns": warns, "info": info}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", default=os.path.join(BASE, "results", "grid"))
    ap.add_argument("--estado", default=os.path.join(BASE, "results", "checagem_estado.json"))
    ap.add_argument("--log", action="store_true", help="anexa o resumo a logs/checagem.log")
    ap.add_argument("--all", action="store_true", help="rechecar tudo")
    ap.add_argument("--sem-modelo", action="store_true", help="pula o teste de determinismo")
    ap.add_argument("--max", type=int, default=0, help="checar no maximo N execucoes novas (0 = todas)")
    a = ap.parse_args()
    estado = {} if a.all or not os.path.exists(a.estado) else json.load(open(a.estado))
    runs = sorted(d for d in glob.glob(os.path.join(a.grid_root, "*", "*", "metrics.json")) if "_superseded" not in d)
    novos = [os.path.dirname(r) for r in runs if os.path.relpath(os.path.dirname(r), a.grid_root) not in estado
             or estado[os.path.relpath(os.path.dirname(r), a.grid_root)]["status"] == "PENDENTE"]
    if a.max:
        novos = novos[:a.max]
    t0 = time.time()
    linhas, n_fail, n_pend = [], 0, 0
    for d in novos:
        r = checar(d, com_modelo=not a.sem_modelo)
        rel = os.path.relpath(d, a.grid_root)
        estado[rel] = {"status": r["status"], "checked_at": datetime.now().isoformat(timespec="seconds"), "fails": r["fails"], "warns": r["warns"], "info": r["info"]}
        n_fail += r["status"] == "FAIL"; n_pend += r["status"] == "PENDENTE"
        if r["status"] != "OK":
            i = r["info"]
            linhas.append(f"  {r['status']:8s} {rel} [{i.get('model')} {i.get('symbol')} H={i.get('H')} {i.get('norm')} keras={i.get('keras')}] " + "; ".join(r["fails"] + r["warns"]))
    os.makedirs(os.path.dirname(a.estado), exist_ok=True)
    json.dump(estado, open(a.estado, "w"), indent=1)
    tot_fail = sum(1 for v in estado.values() if v["status"] == "FAIL")
    resumo = (f"[CHECAGEM] {datetime.now().isoformat(timespec='seconds')} novas={len(novos)} ok={len(novos) - n_fail - n_pend} "
              f"FAIL={n_fail} pendentes_reexecucao={n_pend} | acumulado: {len(estado)} checadas, {tot_fail} FAIL | {time.time() - t0:.0f}s")
    print(resumo)
    for l in linhas:
        print(l)
    if a.log:
        os.makedirs(os.path.join(BASE, "logs"), exist_ok=True)
        with open(os.path.join(BASE, "logs", "checagem.log"), "a") as f:
            f.write(resumo + "\n" + "".join(l + "\n" for l in linhas))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
