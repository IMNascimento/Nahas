#!/usr/bin/env python
"""
Matriz 2x2 que quantifica o confundimento entre normalização e features.

Pergunta
--------
Quanto da vantagem reportada da normalização local vinha, de fato, de o braço
local receber a própria série do alvo como canal extra de X?

    A  global + legacy      (4 canais)  -> reproduz o numero atual
    B  global + equalized   (5 canais)  -> e se o global tambem tiver o close?
    C  local  + equalized   (5 canais)  -> numero atual do local
    D  local  + never       (4 canais)  -> e se o local perder o close?

A distância A→B e a distância C→D medem o mesmo efeito por dois lados.

Uso
---
    cd src
    ./venv/bin/python experiments/run_confound_matrix.py --dry-run
    ./venv/bin/python experiments/run_confound_matrix.py \
        --base-config results/train/b0d13639/hiperparams/config_BTCUSDT_1h_USDT_BIN_b0d13639.json \
        --framework keras --model-type lstm --seeds 1

Requer banco acessível (o treino lê de PriceHistory).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARMS = [
    # (rotulo, strategy, include_target_channel, x_mode, y_mode)
    ("A_global_legacy",    "global", "legacy",    None,     None),
    ("B_global_equalized", "global", "equalized", None,     None),
    ("C_local_equalized",  "local",  "equalized", "minmax", "minmax_target"),
    ("D_local_never",      "local",  "never",     "minmax", "minmax_target"),
]


def build_config(base: dict, arm, seed: int) -> dict:
    label, strategy, include, x_mode, y_mode = arm
    cfg = copy.deepcopy(base)

    norm = {"strategy": strategy}
    if strategy == "global":
        norm["scaler_type"] = base.get("normalization", {}).get("scaler_type", "minmax")
    else:
        norm["x_mode"] = x_mode
        norm["y_mode"] = y_mode

    cfg["normalization"] = norm
    cfg["include_target_channel"] = include
    if include == "legacy":
        cfg["embargo"] = 0            # legacy reproduz o split antigo
    else:
        cfg.pop("embargo", None)      # padrao = window_size + steps_ahead - 1
    cfg["seed"] = int(seed)
    cfg.pop("run_id", None)
    return cfg


def summarize(label: str, seed: int, metrics: dict) -> dict:
    test = metrics.get("test", {})
    setup = metrics.get("setup", {})
    b = metrics.get("baselines") or {}
    persist = (b.get("persistence") or {})
    mim = (b.get("mimicry") or {})
    dm = (b.get("dm_model_vs_persistence") or {})
    summ = (b.get("summary") or {})
    return {
        "arm": label,
        "seed": seed,
        "strategy": setup.get("normalization_strategy"),
        "channels": setup.get("n_channels"),
        "target_in_X": setup.get("target_channel_in_X"),
        "embargo": setup.get("embargo"),
        "rmse": test.get("rmse"),
        "mae": test.get("mae"),
        "r2": test.get("r2"),
        "rmse_persist": persist.get("rmse"),
        "excess_pct": summ.get("excess_rmse_over_persistence_pct"),
        "dm": dm.get("dm_stat"),
        "corr_returns": mim.get("corr_returns"),
        "dir_acc": mim.get("directional_accuracy"),
    }


def fmt_table(rows) -> str:
    cols = [
        ("arm", "braco", 20), ("seed", "seed", 6), ("channels", "can", 4),
        ("target_in_X", "alvo_X", 7), ("rmse", "RMSE", 12), ("r2", "R2", 8),
        ("rmse_persist", "RMSE_pers", 12), ("excess_pct", "excesso%", 10),
        ("dm", "DM", 8), ("corr_returns", "corr_ret", 9),
    ]
    head = " ".join(f"{h:>{w}}" for _, h, w in cols)
    out = [head, "-" * len(head)]
    for r in rows:
        cells = []
        for key, _, w in cols:
            v = r.get(key)
            if v is None:
                s = "-"
            elif isinstance(v, float):
                s = f"{v:.4f}" if abs(v) < 1e4 else f"{v:.1f}"
            else:
                s = str(v)
            cells.append(f"{s:>{w}}")
        out.append(" ".join(cells))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True, help="JSON de um run existente")
    ap.add_argument("--framework", default="keras")
    ap.add_argument("--model-type", default="lstm")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--first-seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None, help="sobrescreve epochs")
    ap.add_argument("--out", default="results/confound_matrix.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="mostra os configs sem treinar (nao precisa de banco)")
    args = ap.parse_args()

    with open(args.base_config, encoding="utf-8") as f:
        base = json.load(f)
    if args.epochs:
        base["epochs"] = args.epochs

    seeds = [args.first_seed + i for i in range(args.seeds)]

    if args.dry_run:
        for arm in ARMS:
            cfg = build_config(base, arm, seeds[0])
            print(f"--- {arm[0]} ---")
            print(json.dumps(
                {k: cfg[k] for k in ("normalization", "include_target_channel", "seed")
                 if k in cfg} | {"embargo": cfg.get("embargo", "padrao")},
                indent=2, ensure_ascii=False))
        print(f"\n{len(ARMS)} bracos x {len(seeds)} seed(s) = {len(ARMS)*len(seeds)} execucoes")
        return 0

    from services.model_service import ModelService  # import tardio: puxa TF/banco

    service = ModelService()
    rows, failures = [], []

    for seed in seeds:
        for arm in ARMS:
            label = arm[0]
            cfg = build_config(base, arm, seed)
            print(f"\n{'='*70}\n>>> {label} | seed={seed}\n{'='*70}")
            t0 = time.time()
            try:
                res = service.train(cfg, args.framework, args.model_type)
                row = summarize(label, seed, res["metrics"])
                row["run_id"] = res["run_id"]
                row["minutes"] = round((time.time() - t0) / 60, 1)
                rows.append(row)
                print(f"<<< {label}: RMSE={row['rmse']}, excesso={row['excess_pct']}%, "
                      f"({row['minutes']} min)")
            except Exception as e:
                failures.append({"arm": label, "seed": seed, "error": str(e)})
                print(f"<<< {label} FALHOU: {e}")
                traceback.print_exc()

    print("\n\n" + "=" * 70)
    print("MATRIZ DO CONFUNDIMENTO")
    print("=" * 70)
    print(fmt_table(rows))

    by_arm = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r["rmse"])

    def med(label):
        vals = [v for v in by_arm.get(label, []) if v is not None]
        return sum(vals) / len(vals) if vals else None

    a, b, c, dd = (med(x[0]) for x in ARMS)
    print("\nLEITURA")
    if a and b and c:
        fechou = (a - b) / (a - c) * 100 if abs(a - c) > 1e-12 else float("nan")
        print(f"  A(global legacy)={a:.4f}  B(global+close)={b:.4f}  C(local)={c:.4f}")
        print(f"  Adicionar o canal do alvo ao global fecha {fechou:.1f}% da distancia ate o local.")
        print("  -> quanto mais perto de 100%, mais a 'vantagem da normalizacao local'")
        print("     era, na verdade, o canal extra de feature.")
    if c and dd:
        print(f"  C(local com close)={c:.4f}  D(local sem close)={dd:.4f}  "
              f"degradacao={100*(dd-c)/c:+.1f}%")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "failures": failures}, f, indent=2, ensure_ascii=False)
    print(f"\nResumo salvo em {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
