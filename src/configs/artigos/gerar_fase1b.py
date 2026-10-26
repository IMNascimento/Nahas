#!/usr/bin/env python
"""
Gera os configs da segunda leva da fase 1 (26/10/2026), todos derivados de
artigo1_principal_H1.json (mesmos hiperparametros, mesma particao 70/15/15):

  artigo1_rolling_<ATIVO>_o<k>.json  origem rolante do artigo 1: 5 origens, teste de
                                     90 dias (2160 h), janela de treino expansivel;
                                     LSTM e Transformer x {global Z, janela Z, ancorada}
                                     (6 execucoes por config, 20 configs = 120)
  artigo4_L<L>_H1[_linear|_evomsn]   varredura de L in {24, 48, 168} com 5 sementes
                                     (300 por L; a varredura de L do artigo 1 e um subconjunto)
  artigo4_ablacao_A{1,2,3}.json      ablacoes do artigo 4 (30 cada)
  artigo4_sementes_H24[_linear|_evomsn]  replica em H=24 das 4 sementes extras (240)

Sementes: principal 1482563973 + default_rng(1482563973).integers(1, 2**31-1, 4).
    ./venv/bin/python configs/artigos/gerar_fase1b.py
"""
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from optimization.rolling_origin import build_origin_configs  # noqa: E402

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = json.load(open(os.path.join(AQUI, "artigo1_principal_H1.json")))
SEED_PRINCIPAL = 1482563973
SEEDS_EXTRA = [2026382952, 1710911661, 1295955485, 1052860402]
SEEDS5 = [SEED_PRINCIPAL] + SEEDS_EXTRA
ATIVOS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
# primeira vela 1h de cada par no banco (confere com a provenance das grades)
INICIO = {"BTCUSDT": "2017-08-18 00:00:00", "ETHUSDT": "2017-08-18 00:00:00",
          "SOLUSDT": "2020-08-11 06:00:00", "BNBUSDT": "2017-11-06 03:00:00"}
NORMS = BASE["NORMALIZATION"]
GLOBAL, WINZ, WINMM, ANCH, EVO = NORMS
NAO_EVO = [GLOBAL, WINZ, WINMM, ANCH]


def novo(grid_id, descricao, **campos):
    cfg = copy.deepcopy(BASE)
    cfg["grid_id"] = grid_id
    cfg["_descricao"] = descricao
    for k, v in campos.items():
        cfg[k] = v
    return cfg


def salvar(cfg):
    p = os.path.join(AQUI, cfg["grid_id"] + ".json")
    json.dump(cfg, open(p, "w"), indent=2, ensure_ascii=False)
    n = 1
    for k, v in cfg.items():
        if isinstance(v, list) and not k.startswith("_"):
            n *= len(v)
    print(f"{cfg['grid_id']:40s} {n:4d} execucoes")
    return n


total = 0

# ---------------------------------------------------------------- origem rolante
for ativo in ATIVOS:
    base_ro = {"start_date": INICIO[ativo], "end_date": BASE["END_DATE"][0], "interval": "1h", "run_id": "ro"}
    for c in build_origin_configs(base_ro, n_origins=5, test_hours=24 * 90, val_fraction=0.15):
        ro = c["rolling_origin"]
        total += salvar(novo(
            f"artigo1_rolling_{ativo}_o{ro['k']}",
            f"Artigo 1, origem rolante {ro['k'] + 1}/5 em {ativo}: serie ate {c['end_date']}, teste = ultimas "
            f"2160 h (inicio esperado {ro['test_start_expected']}), validacao 15% do restante, treino expansivel. "
            f"Fracoes calculadas sobre {ro['n_bars_expected']} barras esperadas desde {INICIO[ativo]}. "
            "LSTM e Transformer x {global Z, janela Z, ancorada}; persistencia vem nos baselines.",
            START_DATE=[INICIO[ativo]], END_DATE=[c["end_date"]], TRAIN_SIZE=[c["train_size"]],
            VALIDATION_SPLIT=[c["validation_split"]], SYMBOL=[ativo], model_type=["lstm", "transformer"],
            NORMALIZATION=[GLOBAL, WINZ, ANCH], _rolling_origin=ro))

# ---------------------------------------------------------------- varredura de L (artigo 4; artigo 1 usa subconjunto)
for L in (24, 48, 168):
    d = f"Artigo 4, varredura de L: L={L}, H=1, 5 sementes, 4 ativos"
    total += salvar(novo(f"artigo4_L{L}_H1", d + ", LSTM/Transformer x 4 normalizacoes (160).",
                         WINDOW_SIZE=[L], SEED=SEEDS5, model_type=["lstm", "transformer"], NORMALIZATION=NAO_EVO))
    total += salvar(novo(f"artigo4_L{L}_H1_linear", d + ", linear x 5 normalizacoes (100).",
                         WINDOW_SIZE=[L], SEED=SEEDS5, model_type=["linear"], NORMALIZATION=NORMS))
    total += salvar(novo(f"artigo4_L{L}_H1_evomsn", d + ", LSTM/Transformer x EvoMSN (40).",
                         WINDOW_SIZE=[L], SEED=SEEDS5, model_type=["lstm", "transformer"], NORMALIZATION=[EVO]))

# ---------------------------------------------------------------- ablacoes (artigo 4)
abl = dict(SEED=SEEDS5, SYMBOL=["BTCUSDT", "ETHUSDT"], NORMALIZATION=[GLOBAL, WINZ, ANCH])
total += salvar(novo("artigo4_ablacao_A1", "Artigo 4, A1: Transformer sem LayerNorm nos blocos (residuais mantidos); "
                     "3 normalizacoes x 5 sementes x BTC/ETH, L=96, H=1 (30). Referencia sem ablacao = artigo1_sementes_H1 + principal.",
                     model_type=["transformer"], USE_LAYERNORM=[False], **abl))
total += salvar(novo("artigo4_ablacao_A2", "Artigo 4, A2: Transformer com codificacao posicional senoidal; 3 normalizacoes x 5 sementes x BTC/ETH (30).",
                     model_type=["transformer"], POSITIONAL_ENCODING=["sinusoidal"], **abl))
total += salvar(novo("artigo4_ablacao_A3", "Artigo 4, A3: LSTM com LayerNorm apos cada camada recorrente; 3 normalizacoes x 5 sementes x BTC/ETH (30).",
                     model_type=["lstm"], LAYER_NORM=[True], **abl))

# ---------------------------------------------------------------- replica H=24 com sementes extras (artigo 4)
d = "Artigo 4, replica em H=24 das 4 sementes extras (a semente principal ja esta em artigo1_principal_H24), L=96"
total += salvar(novo("artigo4_sementes_H24", d + ", LSTM/Transformer x 4 normalizacoes (128).",
                     STEPS_AHEAD=[24], SEED=SEEDS_EXTRA, model_type=["lstm", "transformer"], NORMALIZATION=NAO_EVO))
total += salvar(novo("artigo4_sementes_H24_linear", d + ", linear x 5 normalizacoes (80).",
                     STEPS_AHEAD=[24], SEED=SEEDS_EXTRA, model_type=["linear"], NORMALIZATION=NORMS))
total += salvar(novo("artigo4_sementes_H24_evomsn", d + ", LSTM/Transformer x EvoMSN (32).",
                     STEPS_AHEAD=[24], SEED=SEEDS_EXTRA, model_type=["lstm", "transformer"], NORMALIZATION=[EVO]))
print("total", total)
