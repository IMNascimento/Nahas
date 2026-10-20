# optimization/rolling_origin.py
"""
Avaliação com origem rolante (Tashman, 2000): o mesmo config é treinado várias
vezes, cada vez com a série cortada numa data de origem diferente, e o bloco de
teste tem sempre a mesma duração. Serve para verificar se o ordenamento entre
configurações depende do período de teste escolhido.

Desenho
-------
  origem k (k = 0..K-1):   dados em [start_date, end_k]
                            end_k = end_date − (K−1−k)·test_hours
                            teste = últimas `test_hours` barras antes de end_k
                            validação = `val_fraction` do que sobra
                            treino = o restante (janela expansível)

O serviço de treino particiona por fração, por isso cada origem recebe
`train_size` e `validation_split` recalculados a partir do número esperado de
barras no intervalo. A contagem assume a frequência declarada em `interval`
(ex.: "1h") e série contínua; lacunas pequenas da exchange deslocam a fronteira
em poucas barras, o que é registrado em `n_bars_expected` para auditoria.

Uso
---
    from optimization.rolling_origin import build_origin_configs, run_rolling_origin
    cfgs = build_origin_configs(config, n_origins=5, test_hours=24*90)
    relatorio = run_rolling_origin(service, config, "keras", "lstm", n_origins=5, test_hours=24*90)
"""
from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

_FMT = "%Y-%m-%d %H:%M:%S"


def _parse(dt: str) -> datetime:
    return datetime.strptime(str(dt)[:19], _FMT)


def _bars_between(start: datetime, end: datetime, interval: str) -> int:
    step = pd.Timedelta(interval if any(c.isalpha() for c in interval) else f"{interval}h")
    return int((end - start) / step) + 1


def build_origin_configs(
    config: Dict[str, Any],
    *,
    n_origins: int,
    test_hours: int,
    val_fraction: float = 0.15,
    interval: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Gera uma lista de configs, uma por origem, com `end_date`, `train_size` e
    `validation_split` ajustados para que o bloco de teste tenha `test_hours`
    barras em todas as origens. Não treina nada.
    """
    if n_origins < 1:
        raise ValueError("n_origins deve ser >= 1.")
    if test_hours < 1:
        raise ValueError("test_hours deve ser >= 1.")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("val_fraction deve estar em (0, 1).")

    interval = interval or config.get("interval", "1h")
    start = _parse(config["start_date"])
    end_full = _parse(config["end_date"])
    step = pd.Timedelta(interval if any(c.isalpha() for c in interval) else f"{interval}h")

    out: List[Dict[str, Any]] = []
    for k in range(n_origins):
        end_k = end_full - (n_origins - 1 - k) * test_hours * step
        n_bars = _bars_between(start, end_k, interval)
        if n_bars <= test_hours * 2:
            raise ValueError(
                f"Origem {k}: apenas {n_bars} barras entre {start} e {end_k}; "
                f"insuficiente para um teste de {test_hours} barras."
            )
        test_frac = test_hours / n_bars
        rest = 1.0 - test_frac
        val_frac = rest * val_fraction
        train_frac = rest - val_frac

        cfg = copy.deepcopy(config)
        cfg["end_date"] = end_k.strftime(_FMT)
        cfg["train_size"] = round(train_frac, 6)
        cfg["validation_split"] = round(val_frac, 6)
        cfg["rolling_origin"] = {
            "k": k,
            "n_origins": n_origins,
            "test_hours": test_hours,
            "test_start_expected": (end_k - (test_hours - 1) * step).strftime(_FMT),
            "n_bars_expected": n_bars,
        }
        base_id = str(config.get("run_id") or "ro")
        cfg["run_id"] = f"{base_id}_o{k}"
        out.append(cfg)
    return out


def run_rolling_origin(
    service,
    config: Dict[str, Any],
    framework: str,
    model_type: str,
    *,
    n_origins: int,
    test_hours: int,
    val_fraction: float = 0.15,
    report_path: Optional[str] = None,
    **train_kwargs,
) -> Dict[str, Any]:
    """
    Treina o config em cada origem via `service.train(...)` e consolida as
    métricas de teste e o relatório de baselines de cada execução.
    `train_kwargs` são repassados ao `service.train` (symbol, interval, source...).
    """
    cfgs = build_origin_configs(
        config, n_origins=n_origins, test_hours=test_hours, val_fraction=val_fraction
    )
    linhas: List[Dict[str, Any]] = []
    for cfg in cfgs:
        res = service.train(cfg, framework, model_type, run_id=cfg["run_id"], **train_kwargs) or {}
        metrics = res.get("metrics", {}) if isinstance(res, dict) else {}
        linhas.append({
            "run_id": cfg["run_id"],
            **cfg["rolling_origin"],
            "end_date": cfg["end_date"],
            "test": metrics.get("test"),
            "baselines": metrics.get("baselines"),
        })
    relatorio = {
        "framework": framework,
        "model_type": model_type,
        "normalization": config.get("normalization"),
        "origins": linhas,
    }
    if report_path:
        os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(relatorio, f, indent=2, ensure_ascii=False, default=str)
    return relatorio
