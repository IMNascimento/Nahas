#!/usr/bin/env python
"""
Simulacao de fluxo continuo do artigo 5 (Algoritmo 1 do texto), ligada ao Nahas.

    python run_streaming.py configs/artigos/artigo5_fluxo.json
    python run_streaming.py configs/artigos/artigo5_fluxo.json --only symbol=BTCUSDT,model=lstm,seed=1482563973
    python run_streaming.py configs/artigos/artigo5_fluxo.json --workers 4 --gpus 0,1
    python run_streaming.py configs/artigos/artigo5_fluxo.json --summary-only

Para cada (ativo, backbone, semente):
  1. treina o modelo inicial uma vez por normalizacao (global Z e ancorada) com
     `ModelService.train` sobre os dados ate `init_end` (70/15/15; a ultima
     particao e o periodo de validacao do monitor), em
     results/streaming/<grid>/init_<ativo>_<modelo>_<norm>_<semente>/;
  2. calibra o CUSUM (k, ARL0) sobre z = (y − ŷ)/s do periodo de validacao;
  3. roda as politicas (A..E e controles, ver STREAM_POLICIES) hora a hora de
     `sim_start` a `sim_end`, com fine-tuning nos ultimos N dias (V = ultimos
     `finetune_val_days` dias, embargo de L+H−1 horas), rollback se nao melhorar
     em V, e replay do modelo rejeitado no bloco seguinte;
  4. grava, em results/streaming/<grid>/<ativo>_<modelo>_<politica>_<semente>/:
     predictions.csv, blocks.csv, events.json, metrics.json e provenance.json.
Ao final (ou com --summary-only) regenera results/streaming/<grid>/summary.csv.

Idempotente: pula execucoes com metrics.json e reaproveita o modelo inicial e a
calibracao quando ja existem. Cada job (ativo, backbone, semente) roda num
processo proprio com uma GPU fixa (CUDA_VISIBLE_DEVICES), 2 por GPU por padrao.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import glob
import json
import multiprocessing
import os
import sys
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)

from optimization.streaming import (  # noqa: E402
    STREAM_POLICIES, block_bounds, finetune_slices, invert_block, prepare_block, simulate_hourly,
)
from utils.cusum import CusumMonitor, calibrate_threshold, standardized_residuals  # noqa: E402

_FMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_STREAMING = {
    "init_start": "2017-08-18 00:00:00",
    "init_end": "2022-12-31 23:59:59",
    "sim_start": "2023-01-01 00:00:00",
    "sim_end": "2024-11-11 23:59:59",
    "block_hours": 720,
    "period_hours": 720,
    "finetune_days": 90,
    "finetune_val_days": 7,
    "finetune_epochs": 50,
    "finetune_patience": 10,
    "finetune_lr_factor": 0.1,
    "cusum_k": 0.5,
    "arl0": 720,
    "rollback_tolerance_pct": 0.0,
    "policies": list(STREAM_POLICIES),
    "normalizations": {
        "global": {"strategy": "global", "scaler_type": "standard"},
        "anchored": {"strategy": "local", "x_mode": "zscore", "y_mode": "anchored"},
    },
}


# ---------------------------------------------------------------- config
def load_unified(path: str):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    unified = {k: v for k, v in raw.items() if not k.startswith("_")}
    stream = {**DEFAULT_STREAMING, **unified.pop("STREAMING", {})}
    grid_id = unified.get("grid_id") or os.path.splitext(os.path.basename(path))[0]
    unified["grid_id"] = grid_id
    symbols = unified.pop("SYMBOL")
    models = unified.pop("model_type", None) or unified.pop("MODEL_TYPE")
    seeds = unified.pop("SEED")
    unified.pop("NORMALIZATION", None)   # as normalizacoes vem de STREAMING
    return {"grid_id": grid_id, "unified": unified, "stream": stream,
            "symbols": list(symbols) if isinstance(symbols, list) else [symbols],
            "models": list(models) if isinstance(models, list) else [models],
            "seeds": [int(s) for s in (seeds if isinstance(seeds, list) else [seeds])]}


def jobs_from(spec: dict, only: str = ""):
    filtros = dict(kv.split("=") for kv in only.split(",") if kv)
    out = []
    for symbol in spec["symbols"]:
        for model in spec["models"]:
            for seed in spec["seeds"]:
                if filtros.get("symbol", symbol) != symbol or filtros.get("model", model) != model:
                    continue
                if "seed" in filtros and int(filtros["seed"]) != seed:
                    continue
                out.append((symbol, model, seed))
    return out


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------- um job (processo filho)
def _setup_process(gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if gpu is None else str(gpu)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    try:
        from database.model_base import db
        if not db.is_closed():
            db.close()
        db.connect(reuse_if_open=False)
    except Exception as e:  # o banco so e exigido pelo treino inicial
        print(f"[STREAM] aviso: banco indisponivel neste processo: {e}", flush=True)


def _base_config(spec: dict, svc, symbol: str, model: str, seed: int, norm: dict, stream: dict) -> dict:
    """Config em chaves minusculas, no formato que ModelService.train espera."""
    base_cfg, grid = svc._split_unified_grid_config(spec["unified"])
    if grid:
        raise ValueError(f"chaves com varredura nao suportadas aqui: {sorted(grid)} (fixe cada uma num valor)")
    cfg = svc._apply_params_and_seed_unified(base_cfg, {"SYMBOL": symbol, "MODEL_TYPE": model,
                                                         "SEED": seed, "NORMALIZATION": norm})
    cfg["symbol"] = symbol
    cfg["start_date"] = stream["init_start"]
    cfg["end_date"] = stream["init_end"]
    cfg["use_gpu"] = os.environ.get("CUDA_VISIBLE_DEVICES", "") != ""
    cfg["gpu_index"] = 0
    cfg.pop("parallel", None)
    cfg.pop("grid_id", None)
    cfg["include_target_channel"] = cfg.get("include_target_channel", "equalized")
    return cfg


def _init_model(spec, svc, symbol, model, seed, norm_name, norm, stream, grid_root):
    """Treina (ou reaproveita) o modelo inicial e calibra o CUSUM. Devolve dict com caminhos e calibracao."""
    run_id = f"init_{symbol}_{model}_{norm_name}_{seed}"
    init_dir = os.path.join(grid_root, run_id)
    meta_path = os.path.join(init_dir, "streaming_init.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        if os.path.exists(meta["model_path"]):
            print(f"[STREAM] reaproveitando {run_id}", flush=True)
            return meta
    cfg = _base_config(spec, svc, symbol, model, seed, norm, stream)
    cfg["run_id"] = run_id
    t0 = time.time()
    res = svc.train(cfg, cfg.pop("framework", "keras"), model, run_id=run_id,
                    artifacts_base=f"streaming/{spec['grid_id']}", symbol=symbol,
                    interval=cfg.get("interval", "1h"), currency=cfg.get("currency", "USDT"))
    train_s = time.time() - t0
    # periodo de validacao do monitor = particao de teste do treino inicial (CSV com a escala)
    te = pd.read_csv(res["csv_path"])
    te = te[te["step"] == 1] if "step" in te.columns else te
    z_ref = standardized_residuals(te["real_value"].to_numpy(float), te["predicted_value"].to_numpy(float),
                                   te["scale"].to_numpy(float))
    cal = calibrate_threshold(z_ref, k=float(stream["cusum_k"]), target_arl0=float(stream["arl0"]),
                              rng=np.random.default_rng(seed))
    meta = {"run_id": run_id, "model_path": res["model_path"], "scaler_dir": os.path.join(init_dir, "scaler"),
            "config_path": (glob.glob(os.path.join(init_dir, "hiperparams", "config_*.json")) or [None])[0],
            "metrics_path": res["metrics_path"], "val_csv": res["csv_path"], "train_seconds": train_s,
            "cusum": cal, "n_val": int(len(z_ref)),
            "val_period": [str(te["timestamp"].iloc[0]), str(te["timestamp"].iloc[-1])],
            "config": cfg}
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=_jsonable)
    return meta


def _load_prices(symbol, cfg, stream):
    from database.model_nocapital import PriceHistory
    ws, H = int(cfg["window_size"]), int(cfg.get("steps_ahead", 1))
    margem = timedelta(days=int(stream["finetune_days"]) + 2, hours=ws + H)
    ini = (datetime.strptime(stream["sim_start"], _FMT) - margem).strftime(_FMT)
    df = PriceHistory.get_between_dates(start_date=ini, end_date=stream["sim_end"], symbol=symbol,
                                        interval=cfg.get("interval", "1h"), with_meta=True)
    if "currency" in df.columns and cfg.get("currency"):
        df = df[df["currency"] == cfg["currency"]]
    target = cfg["target_column"]
    feats = [c for c in cfg["relevant_columns"] if c != "timestamp"]
    cols = list(dict.fromkeys(feats + [target, "timestamp"]))
    for c in cols:
        if c != "timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[cols].dropna().reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[~df["timestamp"].duplicated(keep="last")].sort_values("timestamp").reset_index(drop=True)
    s0 = int(np.searchsorted(df["timestamp"].to_numpy(), np.datetime64(stream["sim_start"])))
    if s0 < int(stream["finetune_days"]) * 24 + ws + H - 1:
        raise RuntimeError("historico insuficiente antes de sim_start para o primeiro fine-tuning.")
    return df, s0


def _run_policy(policy, meta, cfg, df, s0, stream, out_dir, seed, symbol, model, norm_name):
    import tensorflow as tf
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.models import clone_model, load_model
    from tensorflow.keras.optimizers import Adam
    from data.data_processing import DataProcessor
    from utils.baselines import diebold_mariano, regression_metrics
    from utils.provenance import code_version, data_fingerprint, environment, write_provenance

    norm_kind, trigger = STREAM_POLICIES[policy]
    inverse_kind = "global" if norm_kind == "global" else "local"
    ws, H = int(cfg["window_size"]), int(cfg.get("steps_ahead", 1))
    ctx_n = ws + H - 1
    n_hours = len(df) - s0
    processor = DataProcessor(window_size=ws, scaler_type=cfg["normalization"].get("scaler_type", "standard"))
    if inverse_kind == "global":
        processor.load_scaler(meta["scaler_dir"])
    tf.keras.utils.set_random_seed(int(seed))
    model0 = load_model(meta["model_path"], safe_mode=False)
    lr_ft = float(cfg.get("learning_rate", 1e-3)) * float(stream["finetune_lr_factor"])
    bs = int(cfg.get("batch_size", 32))
    embargo = ctx_n
    gpu_log = []

    def _prep(a, b):
        return prepare_block(cfg, df.iloc[s0 + a - ctx_n: s0 + b], processor, inverse_kind=inverse_kind)

    def predict_fn(state, a, b):
        prep = _prep(a, b)
        y_hat, scale = invert_block(state["model"].predict(prep["Xs"], verbose=0, batch_size=1024), prep,
                                    processor, inverse_kind=inverse_kind)
        return {"y": prep["y"][:, 0], "y_hat": y_hat[:, 0], "y_last": prep["tw"][:, -1, 0], "scale": scale[:, 0]}

    def replay_fn(state, a, b):
        return predict_fn(state, a, b)["y_hat"]

    def finetune_fn(state, t):
        sl = finetune_slices(t, days=int(stream["finetune_days"]), val_days=int(stream["finetune_val_days"]),
                             embargo=embargo)
        tr, va = _prep(*sl["train"]), _prep(*sl["val"])
        cur = state["model"]
        val_cur = float(np.sqrt(np.mean((cur.predict(va["Xs"], verbose=0, batch_size=1024) - va["ys"]) ** 2)))
        novo = clone_model(cur)
        novo.set_weights(cur.get_weights())
        novo.compile(optimizer=Adam(learning_rate=lr_ft), loss="mean_squared_error")
        t0 = time.time()
        hist = novo.fit(tr["Xs"], tr["ys"], validation_data=(va["Xs"], va["ys"]),
                        epochs=int(stream["finetune_epochs"]), batch_size=bs, verbose=0,
                        callbacks=[EarlyStopping(monitor="val_loss", patience=int(stream["finetune_patience"]),
                                                 restore_best_weights=True)])
        seg = time.time() - t0
        val_novo = float(np.sqrt(np.mean((novo.predict(va["Xs"], verbose=0, batch_size=1024) - va["ys"]) ** 2)))
        gpu_log.append({"t": int(t), "seconds": seg, "epochs": len(hist.history["loss"]),
                        "n_train": int(len(tr["ys"])), "n_val": int(len(va["ys"]))})
        return {"model": novo}, val_novo, val_cur, seg

    cal = meta["cusum"]
    monitor = CusumMonitor(k=cal["k"], h=cal["h"], mu=cal["mu"], sd=cal["sd"])
    t_ini = time.time()
    res = simulate_hourly(trigger, n_hours, predict_fn, finetune_fn, {"model": model0}, monitor=monitor,
                          period=int(stream["period_hours"]), block=int(stream["block_hours"]),
                          rollback_tolerance_pct=float(stream["rollback_tolerance_pct"]), replay_fn=replay_fn)
    elapsed = time.time() - t_ini

    # ---- saidas
    os.makedirs(out_dir, exist_ok=True)
    hrs = res["hours"]
    ts = df["timestamp"].iloc[s0:s0 + n_hours].to_numpy()
    pred = pd.DataFrame({"timestamp": ts, **{k: hrs[k] for k in ("y", "y_hat", "y_last", "scale", "z", "s_plus",
                                                                  "s_minus", "alarm", "model_version", "block")}})
    pred.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)

    ev_ft = [e for e in res["events"] if e["type"] == "finetune"]
    linhas = []
    for b, (a, e) in enumerate(block_bounds(n_hours, int(stream["block_hours"]))):
        y, yh, yl = hrs["y"][a:e], hrs["y_hat"][a:e], hrs["y_last"][a:e]
        m = regression_metrics(y, yh)
        mp = regression_metrics(y, yl)
        fts = [x for x in ev_ft if a <= x["t"] < e]
        linhas.append({"block": b, "start": str(ts[a]), "end": str(ts[e - 1]), "n": int(e - a),
                       "rmse": m["rmse"], "mae": m["mae"], "rmse_persistence": mp["rmse"],
                       "theil_u": m["rmse"] / mp["rmse"] if mp["rmse"] > 0 else np.nan,
                       "mean_abs_z": float(np.nanmean(np.abs(hrs["z"][a:e]))),
                       "n_alarms": int(hrs["alarm"][a:e].sum()),
                       "retrain_attempted": len(fts), "retrain_accepted": sum(1 for x in fts if x["accepted"]),
                       "gpu_minutes": sum(x["seconds"] for x in fts) / 60.0,
                       "model_version_end": int(hrs["model_version"][e - 1])})
    pd.DataFrame(linhas).to_csv(os.path.join(out_dir, "blocks.csv"), index=False)

    eventos = []
    for ev in res["events"]:
        d = dict(ev)
        d["timestamp"] = str(ts[ev["t"]])
        eventos.append(d)
    with open(os.path.join(out_dir, "events.json"), "w") as f:
        json.dump({"events": eventos, "finetune_log": gpu_log}, f, indent=2, default=_jsonable)

    y, yh, yl = hrs["y"], hrs["y_hat"], hrs["y_last"]
    m, mp = regression_metrics(y, yh), regression_metrics(y, yl)
    dm = diebold_mariano(y, yh, yl, h=H)
    rej = [e for e in ev_ft if not e["accepted"] and e.get("replay")]
    metrics = {
        "policy": policy, "normalization": norm_kind, "trigger": trigger, "symbol": symbol, "model": model,
        "seed": int(seed), "n_hours": int(n_hours), "sim_start": str(ts[0]), "sim_end": str(ts[-1]),
        "rmse": m["rmse"], "mae": m["mae"], "rmse_persistence": mp["rmse"], "mae_persistence": mp["mae"],
        "theil_u": m["rmse"] / mp["rmse"], "dm_vs_persistence": dm,
        "mean_abs_z": float(np.nanmean(np.abs(hrs["z"]))), "std_z": float(np.nanstd(hrs["z"])),
        "n_alarms": res["n_alarms"], "n_retrain": res["n_retrain"], "n_accepted": res["n_accepted"],
        "n_rejected": res["n_rejected"], "gpu_minutes": res["gpu_seconds"] / 60.0,
        "rollback_replay": {"n": len(rej),
                            "rmse_rejected_mean": float(np.mean([e["replay"]["rmse_rejected"] for e in rej])) if rej else None,
                            "rmse_kept_mean": float(np.mean([e["replay"]["rmse_kept"] for e in rej])) if rej else None},
        "cusum": cal, "init_run_id": meta["run_id"], "init_train_seconds": meta.get("train_seconds"),
        "simulation_seconds": elapsed, "finetune": {k: stream[k] for k in ("finetune_days", "finetune_val_days",
                                                                          "finetune_epochs", "finetune_patience",
                                                                          "finetune_lr_factor", "rollback_tolerance_pct")},
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=_jsonable)
    write_provenance(os.path.join(out_dir, "provenance.json"),
                     run_id=os.path.basename(out_dir), policy=policy, trigger=trigger, normalization=cfg["normalization"],
                     seed=int(seed), symbol=symbol, model_type=model, config=cfg, streaming=stream,
                     init_run_id=meta["run_id"], init_model_path=meta["model_path"], cusum=cal,
                     code=code_version(), environment=environment(),
                     data=data_fingerprint(df.iloc[s0:s0 + n_hours], cfg["target_column"], symbol=symbol,
                                           interval=cfg.get("interval", "1h"), period=[str(ts[0]), str(ts[-1])]),
                     simulation_seconds=elapsed)
    print(f"[STREAM] {os.path.basename(out_dir)}: U={metrics['theil_u']:.4f} retrain={res['n_retrain']} "
          f"aceitos={res['n_accepted']} alarmes={res['n_alarms']} gpu={metrics['gpu_minutes']:.1f} min "
          f"({elapsed / 60:.1f} min)", flush=True)
    return metrics


def run_job(args):
    spec, symbol, model, seed, gpu = args
    _setup_process(gpu)
    from services.model_service import ModelService
    svc = ModelService()
    stream = spec["stream"]
    grid_root = os.path.join(SRC, "results", "streaming", spec["grid_id"])
    os.makedirs(grid_root, exist_ok=True)
    policies = [p for p in stream["policies"] if p in STREAM_POLICIES]
    feitos = []
    df_cache = {}
    for norm_name, norm in stream["normalizations"].items():
        pols = [p for p in policies if STREAM_POLICIES[p][0] == norm_name]
        pendentes = [p for p in pols if not os.path.exists(os.path.join(grid_root, f"{symbol}_{model}_{p}_{seed}", "metrics.json"))]
        if not pendentes:
            continue
        meta = _init_model(spec, svc, symbol, model, seed, norm_name, norm, stream, grid_root)
        cfg = meta["config"]
        if symbol not in df_cache:
            df_cache[symbol] = _load_prices(symbol, cfg, stream)
        df, s0 = df_cache[symbol]
        for p in pendentes:
            out_dir = os.path.join(grid_root, f"{symbol}_{model}_{p}_{seed}")
            try:
                _run_policy(p, meta, cfg, df, s0, stream, out_dir, seed, symbol, model, norm_name)
                feitos.append(os.path.basename(out_dir))
            except Exception as e:
                print(f"[STREAM][ERRO] {os.path.basename(out_dir)}: {e}", flush=True)
                with open(os.path.join(grid_root, "erros.log"), "a") as f:
                    f.write(f"{datetime.now().isoformat()} {os.path.basename(out_dir)} {e!r}\n")
    return feitos


def write_summary(grid_root: str) -> str:
    linhas = []
    for p in sorted(glob.glob(os.path.join(grid_root, "*", "metrics.json"))):
        if os.path.basename(os.path.dirname(p)).startswith("init_"):
            continue
        with open(p) as f:
            m = json.load(f)
        linhas.append({k: m[k] for k in ("symbol", "model", "policy", "normalization", "trigger", "seed", "rmse", "mae",
                                         "rmse_persistence", "theil_u", "n_alarms", "n_retrain", "n_accepted",
                                         "n_rejected", "gpu_minutes", "n_hours")}
                      | {"dm_stat": m["dm_vs_persistence"]["dm_stat"], "dm_p": m["dm_vs_persistence"]["p_value"],
                         "cusum_h": m["cusum"]["h"], "cusum_arl0": m["cusum"]["arl0"],
                         "replay_rmse_rejected": m["rollback_replay"]["rmse_rejected_mean"],
                         "replay_rmse_kept": m["rollback_replay"]["rmse_kept_mean"], "run_dir": os.path.dirname(p)})
    out = os.path.join(grid_root, "summary.csv")
    pd.DataFrame(linhas).to_csv(out, index=False)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--only", default="", help="filtros: symbol=BTCUSDT,model=lstm,seed=1482563973")
    ap.add_argument("--workers", type=int, default=None, help="processos em paralelo (padrao: 2 por GPU, ou 1 sem GPU)")
    ap.add_argument("--gpus", default=None, help="lista de GPUs, ex.: 0,1 (padrao: todas; vazio = CPU)")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    spec = load_unified(a.config)
    grid_root = os.path.join(SRC, "results", "streaming", spec["grid_id"])
    if a.summary_only:
        print(write_summary(grid_root))
        return 0
    if a.gpus is None:
        try:
            import subprocess
            n = len(subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL).decode().strip().splitlines())
            gpus = list(range(n))
        except Exception:
            gpus = []
    else:
        gpus = [int(g) for g in a.gpus.split(",") if g.strip() != ""]
    workers = a.workers or (2 * len(gpus) if gpus else 1)
    jobs = jobs_from(spec, a.only)
    print(f"[STREAM] grid={spec['grid_id']} jobs={len(jobs)} gpus={gpus or 'cpu'} workers={workers}", flush=True)
    args = [(spec, s, m, sd, (gpus[i % len(gpus)] if gpus else None)) for i, (s, m, sd) in enumerate(jobs)]
    t0 = time.time()
    if workers == 1:
        for x in args:
            run_job(x)
    else:
        ctx = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            for fut in concurrent.futures.as_completed([ex.submit(run_job, x) for x in args]):
                try:
                    fut.result()
                except Exception as e:
                    print(f"[STREAM][ERRO] job falhou: {e}", flush=True)
    print(f"[STREAM] resumo em {write_summary(grid_root)} ({(time.time() - t0) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
