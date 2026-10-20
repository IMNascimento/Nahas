#!/usr/bin/env python
"""
Dispara uma grade unificada a partir de um JSON (configs/artigos/*.json).

    python run_grid.py configs/artigos/artigo1_principal_H1.json
    python run_grid.py configs/artigos/smoke_servidor.json --score rmse --score-on validation
    python run_grid.py configs/artigos/artigo1_principal_H1.json --rerun-failed   (so as que falharam)

Chaves que comecam com "_" (ex.: "_descricao") sao documentacao e sao ignoradas.
Cada execucao grava provenance.json; o grid grava grid_provenance.json.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--score", default="rmse")
    ap.add_argument("--score-on", default="validation")
    ap.add_argument("--rerun-failed", action="store_true",
                    help="reexecuta apenas as combinacoes sem run_id no grid_summary.json (ex.: timeout)")
    a = ap.parse_args()
    with open(a.config, "r", encoding="utf-8") as f:
        unified = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    from services.model_service import ModelService
    t0 = time.time()
    out = ModelService().grid_search_unified(unified, score=a.score, score_on=a.score_on,
                                             rerun_failed=a.rerun_failed)
    print(f"[RUN_GRID] grid_id={out['grid_id']} | {len(out['summary'])} execucoes | "
          f"{(time.time() - t0) / 60:.1f} min | leaderboard={out['leaderboard_csv']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
