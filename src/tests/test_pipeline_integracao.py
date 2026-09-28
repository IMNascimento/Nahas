"""
Integração da pipeline de treino, sem banco de dados e sem GPU.

O acesso ao MySQL é substituído por dublês: a leitura de preços devolve uma série
sintética e a gravação do run vira um registro em memória. O resto do caminho é o
real, incluindo janelamento, partição, normalização, treino, inversão e métricas.

Rodar de dentro de `src/`:
    ./venv/bin/python tests/test_pipeline_integracao.py
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import services.model_service as ms                        # noqa: E402
from services.model_service import ModelService            # noqa: E402

RNG = np.random.default_rng(3)
N_BARRAS = 600
JANELA = 24


def _serie_sintetica(n=N_BARRAS):
    """Preço com tendência, ciclo diário e ruído, mais volume em outra escala."""
    t = np.arange(n)
    close = 100 + 0.02 * t + 2 * np.sin(2 * np.pi * t / 24.0) + RNG.normal(0, 0.3, n)
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="h"),
        "close": close,
        "open": close + RNG.normal(0, 0.1, n),
        "high": close + np.abs(RNG.normal(0, 0.2, n)),
        "low": close - np.abs(RNG.normal(0, 0.2, n)),
        "volume": RNG.uniform(1e3, 5e3, n),
    })


class _PriceHistoryFake:
    @staticmethod
    def get_between_dates(**kwargs):
        return _serie_sintetica()


class _TrainingRunFake:
    registros = []

    @classmethod
    def create(cls, **kwargs):
        cls.registros.append(kwargs)
        return kwargs


def _com_dubles(fn):
    """Troca banco por dublês, executa e devolve tudo ao lugar."""
    price, run, conn = ms.PriceHistory, ms.TrainingRun, ms.ensure_db_connection
    ms.PriceHistory = _PriceHistoryFake
    ms.TrainingRun = _TrainingRunFake
    ms.ensure_db_connection = lambda *a, **k: None
    try:
        return fn()
    finally:
        ms.PriceHistory, ms.TrainingRun, ms.ensure_db_connection = price, run, conn


def _config(normalizacao: dict, run_id: str):
    return {
        "run_id": run_id,
        "symbol": "TESTUSDT", "interval": "1h", "currency": "USDT",
        "start_date": "2024-01-01 00:00:00", "end_date": "2024-03-01 00:00:00",
        "target_column": "close",
        "relevant_columns": ["close", "open", "high", "low", "volume"],
        "window_size": JANELA, "steps_ahead": 1, "output_units": 1,
        "train_size": 0.7, "validation_split": 0.15,
        "layers_config": [8, 4], "activation_functions": ["tanh", "tanh"],
        "dropout": 0.0, "recurrent_dropout": 0.0,
        "epochs": 2, "batch_size": 32, "patience": 2,
        "learning_rate": 1e-3, "optimizer": "adam",
        "loss_fn": "mean_squared_error",
        "use_gpu": False, "seed": 123,
        "normalization": normalizacao,
    }


ESTRATEGIAS = [
    ("global", {"strategy": "global", "scaler_type": "minmax"}),
    ("local", {"strategy": "local", "x_mode": "minmax", "y_mode": "minmax_target"}),
    ("evomsn", {"strategy": "evomsn", "evomsn_k_scales": 2, "evomsn_predictor": "linear"}),
    ("evomsn_like", {"strategy": "evomsn_like", "evomsn_k_scales": 2}),
]


def _treina(nome, normalizacao):
    svc = ModelService()
    cfg = _config(normalizacao, run_id=f"itest_{nome}")
    return _com_dubles(lambda: svc.train(cfg, "keras", "lstm", artifacts_base="itest"))


def test_pipeline_completa_nas_quatro_estrategias():
    """Cada estratégia treina, inverte e mede sem erro, nas três partições."""
    for nome, normalizacao in ESTRATEGIAS:
        out = _treina(nome, normalizacao)
        metrics = out["metrics"]

        for secao in ("train", "validation", "test"):
            assert secao in metrics, f"[{nome}] falta a seção {secao}"
            assert metrics[secao] is not None, f"[{nome}] seção {secao} vazia"
            for chave in ("mse", "rmse", "mae", "r2", "mape"):
                valor = metrics[secao][chave]
                assert np.isfinite(valor), f"[{nome}] {secao}.{chave} = {valor}"

        assert os.path.exists(out["model_path"]), f"[{nome}] modelo não salvo"
        assert _TrainingRunFake.registros, "registro do run não foi criado"
        print(f"OK  {nome:12s} rmse_val={metrics['validation']['rmse']:.4f} "
              f"rmse_test={metrics['test']['rmse']:.4f} "
              f"mape_test={metrics['test']['mape']:.3f}%")


def test_validacao_e_teste_sao_conjuntos_diferentes():
    """A métrica de seleção não pode ser a mesma coisa que a de relato."""
    out = _treina("local_cmp", {"strategy": "local", "x_mode": "minmax",
                                "y_mode": "minmax_target"})
    m = out["metrics"]
    assert m["validation"]["rmse"] != m["test"]["rmse"], \
        "validação e teste deram exatamente o mesmo RMSE: conjuntos provavelmente iguais"
    print(f"OK  validação e teste são distintos: {m['validation']['rmse']:.4f} "
          f"vs {m['test']['rmse']:.4f}")


def test_politica_legacy_reproduz_quatro_canais():
    """A configuração de reprodução da dissertação continua executável."""
    svc = ModelService()
    cfg = _config({"strategy": "local", "x_mode": "minmax", "y_mode": "minmax_target",
                   "evomsn_short_horizon_policy": "legacy",
                   "evomsn_period_scaling": "legacy"}, run_id="itest_legacy")
    cfg["include_target_channel"] = "legacy"
    cfg["embargo"] = 0

    out = _com_dubles(lambda: svc.train(cfg, "keras", "lstm", artifacts_base="itest"))
    assert np.isfinite(out["metrics"]["test"]["rmse"])
    print(f"OK  configuração legacy roda: rmse_test={out['metrics']['test']['rmse']:.4f}")


def test_config_salvo_registra_a_politica_do_canal():
    """M4: o config gravado precisa dizer a politica, senao a inferencia adivinha."""
    import glob
    import json as _json

    out = _treina("politica", {"strategy": "global", "scaler_type": "minmax"})
    cfgs = glob.glob(os.path.join(os.path.dirname(os.path.dirname(out["model_path"])),
                                  "hiperparams", "*.json"))
    assert cfgs, "config do run nao foi salvo"
    salvo = _json.load(open(cfgs[0], encoding="utf-8"))
    assert "include_target_channel" in salvo, "politica ausente do config salvo"
    assert ModelService.policy_from_config(salvo) == salvo["include_target_channel"]
    print(f"OK  config salvo registra include_target_channel={salvo['include_target_channel']!r}")


def _limpa_artefatos():
    from config.settings import BASE_DIR
    alvo = os.path.join(BASE_DIR, "results", "itest")
    if os.path.isdir(alvo):
        shutil.rmtree(alvo)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    try:
        for fn in fns:
            fn()
        print(f"\n{len(fns)} testes de integração passaram.")
    finally:
        _limpa_artefatos()
