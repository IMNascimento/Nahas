# optimization/grid_search.py
"""
Busca em grade de hiperparametros.

Vivia dentro de `ModelService`, que passava de 2.100 linhas. Foi extraido para
manter a divisao por responsabilidade: `ModelService` treina e avalia um modelo;
este modulo orquestra varias execucoes e escolhe uma.

`ModelService` mantem os mesmos nomes publicos, delegando para ca, de modo que
`service.grid_search_unified(...)` e `ModelService.validar_score(...)` continuam
funcionando como antes.

Ponto metodologico preservado: `score_on` vale "validation" por padrao. Escolher
hiperparametro pela metrica do teste transforma o teste em conjunto de selecao e
infla o resultado final.
"""
from __future__ import annotations

import concurrent.futures
import csv
import json
import os
import time
from datetime import datetime

from config.settings import BASE_DIR
from database.model_nahas import GridResult, TrainingRun
from utils.capacity_train import pick_gpu_for_job
from utils.db_utils import ensure_db_connection


SCORES_VALIDOS = ("rmse", "mae", "mse", "r2", "mape")
CONJUNTOS_VALIDOS = ("train", "validation", "test")

def validar_score(score: str, score_on: str) -> None:
    """Valida a metrica e o conjunto usados para ESCOLHER hiperparametro."""
    if score not in SCORES_VALIDOS:
        raise ValueError(f"score deve ser um de {SCORES_VALIDOS}")
    if score_on not in CONJUNTOS_VALIDOS:
        raise ValueError(f"score_on deve ser um de {CONJUNTOS_VALIDOS}")

def extrair_score(run_metrics: dict, score: str, score_on: str):
    """Devolve o valor da metrica, ou None quando ela nao existe.

    Antes a conversao direta para float recebia None e levantava TypeError
    dentro do registro do resultado, derrubando o grid inteiro.
    """
    secao = (run_metrics or {}).get(score_on) or {}
    bruto = secao.get(score)
    if bruto is None:
        return None
    try:
        return float(bruto)
    except (TypeError, ValueError):
        return None

def impasse_no_escalonador(*, pendentes: int, rodando: int, admitiu: bool,
                           tentativas: int, limite: int = 30) -> bool:
    """True quando o escalonador nao tem como progredir.

    Com nada rodando e nada admitido, o laco antigo voltava ao inicio e
    girava em vazio consumindo um nucleo inteiro, sem mensagem.
    """
    if rodando > 0 or admitiu or pendentes <= 0:
        return False
    return tentativas > limite

def run_grid_search(service, unified_config: dict, *, score: str = "rmse",
                        score_on: str = "validation"):
    """Grid search com suporte a execução paralela segura.

    `score_on` e "validation" por padrao: escolher hiperparametro pela metrica
    do teste transforma o teste em conjunto de selecao e infla o resultado
    final. O teste continua disponivel, mas como relato, nao como criterio.
    """
    service.validar_score(score, score_on)

    base_cfg, grid = service._split_unified_grid_config(unified_config)
    grid_id = base_cfg.get("grid_id") or datetime.now().strftime("grid_%Y%m%d_%H%M%S")
    base_cfg["grid_id"] = grid_id

    grid_root = os.path.join(BASE_DIR, "results", "grid", grid_id)
    os.makedirs(grid_root, exist_ok=True)
    with open(os.path.join(grid_root, "unified_config.json"), "w") as f:
        json.dump(unified_config, f, indent=4)

    par = (base_cfg.get("parallel") or {})
    enabled = bool(par.get("enabled", False))
    backend = str(par.get("backend", "process")).lower()
    max_workers_per_gpu = int(par.get("max_workers_per_gpu", 1))
    safety_ratio = float(par.get("safety_ratio", 0.20))
    cpu_workers = int(par.get("cpu_workers", 1))

    def _finalize_and_record(out, params, fw, mt):
        nonlocal best_record, results_summary
        run_metrics = out.get("metrics", {})
        readable = run_metrics.get("readable", {})
        val = service.extrair_score(run_metrics, score, score_on)
        if val is None:
            print(f"[GRID] metrica {score} ausente em {score_on}; combinacao ignorada",
                  flush=True)
            return
        is_better = (val > (best_record or {}).get("score_value", -1e18)) if score == "r2" else (val < (best_record or {}).get("score_value", 1e18))
        
        try:
            ensure_db_connection()
            tr = TrainingRun.get(TrainingRun.run_uuid == out["run_id"])
            GridResult.create(
                training_run=tr, params_json=json.dumps(params, indent=2),
                train_loss=out.get("train_loss"), val_loss=out.get("val_loss"),
                metrics_json=json.dumps(readable, indent=2),
                model_path=out.get("model_path"), csv_metrics_path=out.get("csv_path"), epoch=None
            )
        except Exception as e:
            print(f"[GridResult] WARN banco: {e}", flush=True)
        
        rec = {
            "run_id": out["run_id"], "params": params, "framework": fw, "model_type": mt,
            "score_on": score_on, "score_name": score, "score_value": val,
            "model_path": out.get("model_path"), "csv_path": out.get("csv_path"),
            "metrics_path": out.get("metrics_path"), "metrics_readable": readable,
        }
        results_summary.append(rec)
        if is_better:
            best_record = rec

    results_summary, best_record = [], None
    base_cfg.setdefault("use_gpu", False)
    base_cfg.setdefault("gpu_index", None)

    combos = list(service._cartesian_product(grid or {}))
    print(f"\n[GRID] Total: {len(combos)} combinações", flush=True)

    if not enabled:
        print("[GRID] Modo SEQUENCIAL", flush=True)
        for i, params in enumerate(combos, 1):
            print(f"\n[GRID] Combo {i}/{len(combos)}", flush=True)
            cfg = service._apply_params_and_seed_unified(base_cfg, params)
            fw = cfg.pop("framework", base_cfg.get("framework"))
            mt = cfg.pop("model_type", base_cfg.get("model_type"))
            if fw is None or mt is None:
                raise ValueError("framework/model_type obrigatórios")
            out = service.train(cfg, fw, mt, artifacts_base=f"grid/{grid_id}")
            _finalize_and_record(out, params, fw, mt)
    else:
        print(f"[GRID] Modo PARALELO: {max_workers_per_gpu} jobs/GPU", flush=True)
        
        try:
            from utils.capacity_train import gpu_mem_info
            gpus = gpu_mem_info()
            num_gpus = len(gpus)
            print(f"[GRID] GPUs: {num_gpus}", flush=True)
        except Exception as e:
            print(f"[GRID] Falha GPU detection: {e}", flush=True)
            num_gpus = 0
        
        gpu_slots_in_use = [0] * max(1, num_gpus)
        pool_size = max(1, cpu_workers + (num_gpus * max_workers_per_gpu))
        print(f"[GRID] Pool: {pool_size} workers", flush=True)

        pending = combos[:]
        running = {}
        completed = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=pool_size) as ex:
            tentativas_sem_progresso = 0
            while pending or running:
                # Admitir novos
                admitiu = False
                i = 0
                while i < len(pending):
                    params = pending[i]
                    cfg = service._apply_params_and_seed_unified(base_cfg, params)
                    fw = cfg.pop("framework", base_cfg.get("framework"))
                    mt = cfg.pop("model_type", base_cfg.get("model_type"))
                    
                    if fw is None or mt is None:
                        raise ValueError("framework/model_type obrigatórios")

                    use_gpu = bool(cfg.get("use_gpu", False))
                    assigned_gpu = None
                    
                    if use_gpu and num_gpus > 0:
                        assigned_gpu = pick_gpu_for_job(cfg, mt, max_workers_per_gpu, gpu_slots_in_use, safety_ratio=safety_ratio)
                        if assigned_gpu is None:
                            i += 1
                            continue
                    else:
                        num_cpu_running = sum(1 for meta in running.values() if meta["gpu_idx"] is None)
                        if num_cpu_running >= cpu_workers:
                            i += 1
                            continue

                    if assigned_gpu is not None:
                        gpu_slots_in_use[assigned_gpu] += 1
                        print(f"[GRID] GPU {assigned_gpu}: {gpu_slots_in_use[assigned_gpu]}/{max_workers_per_gpu} slots", flush=True)

                    args = (cfg, fw, mt, f"grid/{grid_id}", assigned_gpu)
                    fut = ex.submit(_train_job_worker, args)
                    running[fut] = {"params": params, "gpu_idx": assigned_gpu, "framework": fw, "model_type": mt, "combo_idx": len(combos) - len(pending) + 1}
                    pending.pop(i)
                    admitiu = True
                    print(f"[GRID] Submetido: combo {running[fut]['combo_idx']}/{len(combos)} | GPU: {assigned_gpu}", flush=True)

                # Colher finalizados
                if not running:
                    # Nada rodando e nada admitido: sem o guarda abaixo o laco
                    # girava em vazio consumindo um nucleo inteiro, em silencio.
                    tentativas_sem_progresso = 0 if admitiu else tentativas_sem_progresso + 1
                    if service.impasse_no_escalonador(pendentes=len(pending), rodando=0,
                                                   admitiu=admitiu,
                                                   tentativas=tentativas_sem_progresso):
                        raise RuntimeError(
                            f"Nenhuma das {len(pending)} combinacoes restantes cabe nos "
                            f"recursos disponiveis. Revise cpu_workers, "
                            f"max_workers_per_gpu e safety_ratio."
                        )
                    if not admitiu:
                        time.sleep(2.0)
                    continue
                tentativas_sem_progresso = 0
                    
                done, _ = concurrent.futures.wait(running.keys(), timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED)
                
                for fut in done:
                    meta = running.pop(fut)
                    completed += 1
                    
                    if meta["gpu_idx"] is not None:
                        gpu_slots_in_use[meta["gpu_idx"]] = max(0, gpu_slots_in_use[meta["gpu_idx"]] - 1)
                        print(f"[GRID] GPU {meta['gpu_idx']} liberada: {gpu_slots_in_use[meta['gpu_idx']]} slots", flush=True)
                    
                    try:
                        out = fut.result(timeout=5)
                        print(f"[GRID] ✓ Combo {meta['combo_idx']}/{len(combos)} OK", flush=True)
                        _finalize_and_record(out, meta["params"], meta["framework"], meta["model_type"])
                    except Exception as e:
                        print(f"[GRID] ✗ Combo {meta['combo_idx']}/{len(combos)} FALHOU: {e}", flush=True)
                        rec = {
                            "run_id": None, "params": meta["params"], "framework": meta["framework"], "model_type": meta["model_type"],
                            "score_on": score_on, "score_name": score,
                            "score_value": float("inf") if score != "r2" else float("-inf"),
                            "model_path": None, "csv_path": None, "metrics_path": None,
                            "metrics_readable": {"error": str(e)},
                        }
                        results_summary.append(rec)
                
                if done:
                    print(f"[GRID] Progresso: {completed}/{len(combos)} | {len(running)} rodando | {len(pending)} pendentes", flush=True)

    # Salvar resultados
    print(f"\n[GRID] Salvando resultados...", flush=True)
    
    summary_path = os.path.join(grid_root, "grid_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"grid_id": grid_id, "score": score, "score_on": score_on, "best": best_record, "summary": results_summary}, f, indent=4)

    lb_csv = os.path.join(grid_root, "leaderboard.csv")
    with open(lb_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id","framework","model_type","score_on","score_name","score_value","model_path","csv_path","metrics_path","params_json"])
        for rec in sorted(results_summary, key=lambda x: x["score_value"], reverse=(score=="r2")):
            w.writerow([rec.get("run_id"), rec["framework"], rec["model_type"], rec["score_on"], rec["score_name"], rec["score_value"],
                rec.get("model_path"), rec.get("csv_path"), rec.get("metrics_path"), json.dumps(rec["params"])])

    print(f"[GRID] ✓ Concluído! Melhor: {best_record['score_value'] if best_record else 'N/A'}", flush=True)
    
    return {"grid_id": grid_id, "best": best_record, "summary": results_summary, "leaderboard_csv": lb_csv, "summary_json": summary_path}
