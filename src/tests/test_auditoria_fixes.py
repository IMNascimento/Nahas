"""
Testes das correções levantadas na auditoria de pré-publicação.

Cada teste nomeia o achado correspondente do relatório de auditoria.
Rodar de dentro de `src/`:

    ./venv/bin/python -m pytest tests/test_auditoria_fixes.py -q
    ./venv/bin/python tests/test_auditoria_fixes.py           # sem pytest
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.evomsn_normalizer import EvoMSNNormalizer        # noqa: E402
from services.model_service import ModelService            # noqa: E402
from services.trainer_factory import TrainerFactory        # noqa: E402

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RNG = np.random.default_rng(11)


# --------------------------------------------------------------------- A4
def test_metricas_incluem_mape():
    """A4: a dissertação reporta MAPE; o cálculo precisa existir na pipeline."""
    y_true = np.array([[100.0], [200.0], [300.0]])
    y_pred = np.array([[110.0], [180.0], [330.0]])

    m = ModelService._compute_basic_metrics(y_true, y_pred)

    assert "mape" in m, "MAPE ausente nas métricas"
    # |10|/100 + |20|/200 + |30|/300 = 10% em todos -> media 10%
    assert abs(m["mape"] - 10.0) < 1e-9, f"MAPE={m['mape']}"
    assert abs(m["mae"] - 20.0) < 1e-9
    assert abs(m["rmse"] - np.sqrt((100 + 400 + 900) / 3)) < 1e-9
    print(f"OK  MAPE calculado: {m['mape']:.2f}%")


def test_mape_nao_divide_por_zero():
    """A4: série com zero não pode gerar inf nem NaN."""
    m = ModelService._compute_basic_metrics(
        np.array([[0.0], [10.0]]), np.array([[0.0], [11.0]])
    )
    assert np.isfinite(m["mape"]), f"MAPE não finito: {m['mape']}"
    print(f"OK  MAPE finito com alvo zero: {m['mape']:.2f}%")


# --------------------------------------------------------------------- A2
def _janelas_com_dois_ciclos(L=96, n=60):
    """Canal 0 (alvo): ciclo de 24 barras, amplitude 1, quase sem ruído.
       Canal 1 (volume): ciclo de 8 barras, amplitude 10.000, ruído de desvio 300.

    Em amplitude bruta, até o ruído do volume é maior que o ciclo do alvo, que é
    exatamente a situação em que a média de amplitudes entre canais escolhe os
    períodos errados."""
    t = np.arange(L)
    alvo = np.sin(2 * np.pi * t / 24.0)
    volume = 1e4 * np.sin(2 * np.pi * t / 8.0)
    X = np.zeros((n, L, 2))
    for i in range(n):
        X[i, :, 0] = 100 + alvo + RNG.normal(0, 1e-3, L)
        X[i, :, 1] = 5e4 + volume + RNG.normal(0, 300.0, L)
    return X


def test_fft_legacy_e_dominada_pelo_canal_de_maior_escala():
    """A2: comportamento antigo, mantido sob period_scaling='legacy'."""
    X = _janelas_com_dois_ciclos()
    msn = EvoMSNNormalizer(window_size=96, horizon=1, n_features=2,
                           k_scales=2, period_scaling="legacy")
    periodos, _ = msn._select_periods_fft_global(X, 2)

    assert 8 in periodos, f"o ciclo do volume deveria dominar: {periodos}"
    assert 24 not in periodos, f"legacy não deveria enxergar o ciclo do alvo: {periodos}"
    print(f"OK  legacy segue a escala do canal: periodos={periodos}")


def test_fft_normalizada_enxerga_o_ciclo_do_alvo():
    """A2: padronizar por canal antes da FFT devolve o ciclo do alvo."""
    X = _janelas_com_dois_ciclos()
    msn = EvoMSNNormalizer(window_size=96, horizon=1, n_features=2, k_scales=2)
    assert msn.period_scaling == "per_channel", "o padrão deve ser a versão corrigida"

    periodos, _ = msn._select_periods_fft_global(X, 2)

    assert 24 in periodos, f"o ciclo do alvo precisa aparecer: {periodos}"
    assert 8 in periodos, f"o ciclo do volume continua legítimo: {periodos}"
    print(f"OK  per_channel enxerga os dois ciclos: periodos={periodos}")


def test_config_liga_a_escala_de_periodos():
    """A2: a chave de configuração precisa chegar ao normalizador."""
    svc = ModelService()
    _, _, _, _, ev = svc._norm_cfg({"normalization": {"strategy": "evomsn"}})
    assert ev["period_scaling"] == "per_channel", ev

    _, _, _, _, ev = svc._norm_cfg(
        {"normalization": {"strategy": "evomsn", "evomsn_period_scaling": "legacy"}}
    )
    assert ev["period_scaling"] == "legacy", ev
    print("OK  evomsn_period_scaling chega ao normalizador")


def _fixture_msn(L=48, C=3, n=120):
    t = np.arange(L + n)
    preco = 100 + np.cumsum(RNG.normal(0, 0.5, L + n)) + 3 * np.sin(2 * np.pi * t / 24.0)
    X = np.zeros((n, L, C))
    for i in range(n):
        X[i, :, 0] = preco[i:i + L]
        X[i, :, 1] = preco[i:i + L] + RNG.normal(0, 0.2, L)
        X[i, :, 2] = RNG.uniform(1e3, 5e3, L)
    y = preco[L:L + n].reshape(-1, 1)
    return X, y, X[:, :, [0]]


def test_evomsn_e_like_coincidem_sob_horizonte_unitario():
    """Com H = 1 e window_stats, as duas variantes viram a mesma coisa.

    Todas as escalas sao degeneradas, entao a referencia vem da janela do alvo em
    todas elas e os preditores auxiliares, unica diferenca entre as classes, nunca
    sao chamados. Documentado porque a coincidencia exata dos numeros, observada no
    pre-teste com dados reais, e consequencia do desenho, nao erro.
    """
    from data.evomsn_normalizer import EvoMSNLikeNormalizer

    X, y, tw = _fixture_msn()
    comum = dict(window_size=X.shape[1], horizon=1, n_features=X.shape[2],
                 target_idx=0, k_scales=2, random_state=7)

    cheio = EvoMSNNormalizer(**comum, short_horizon_policy="window_stats")
    like = EvoMSNLikeNormalizer(**comum, short_horizon_policy="window_stats")
    cheio.fit(X, y, target_windows=tw)
    like.fit(X, y, target_windows=tw)
    _, _, ctx_cheio = cheio.transform(X, y, target_windows=tw)
    _, _, ctx_like = like.transform(X, y, target_windows=tw)

    assert np.allclose(ctx_cheio["phi_hat_stack"], ctx_like["phi_hat_stack"], atol=1e-12)
    assert np.allclose(ctx_cheio["xi_hat_stack"], ctx_like["xi_hat_stack"], atol=1e-12)

    # sob legacy, os preditores voltam a ser usados e as duas divergem
    cheio_l = EvoMSNNormalizer(**comum, short_horizon_policy="legacy")
    like_l = EvoMSNLikeNormalizer(**comum, short_horizon_policy="legacy")
    cheio_l.fit(X, y, target_windows=tw)
    like_l.fit(X, y, target_windows=tw)
    _, _, ctx_cl = cheio_l.transform(X, y, target_windows=tw)
    _, _, ctx_ll = like_l.transform(X, y, target_windows=tw)
    assert not np.allclose(ctx_cl["phi_hat_stack"], ctx_ll["phi_hat_stack"], atol=1e-6), \
        "sob legacy as duas variantes deveriam divergir"
    print("OK  EvoMSN e EvoMSN-like coincidem sob H=1 com window_stats, e divergem sob legacy")


# --------------------------------------------------------------------- M4
def test_config_antigo_assume_politica_legacy():
    """M4: config salvo antes da chave existir descreve um modelo legacy."""
    assert ModelService.policy_from_config({}) == "legacy"
    assert ModelService.policy_from_config({"include_target_channel": None}) == "legacy"
    assert ModelService.policy_from_config({"include_target_channel": "equalized"}) == "equalized"
    assert ModelService.policy_from_config({"include_target_channel": "never"}) == "never"
    print("OK  config sem a chave resolve para legacy")


def test_mensagem_sem_dados_aponta_o_filtro():
    """M10: a causa usual e divergencia de source/exchange, nao ausencia de historico."""
    msg = ModelService.mensagem_sem_dados(
        "BTCUSDT", "1h", "USDT", "BINANCE", "binance_api",
        "2017-08-18 00:00:00", "2024-11-11 23:59:59",
        [("binance", "BINANCE", 79767)],
    )
    assert "source='binance_api'" in msg, msg
    assert "source='binance'" in msg and "79767" in msg, msg

    vazio = ModelService.mensagem_sem_dados(
        "XPTOUSDT", "1h", "USDT", "BINANCE", "binance", "a", "b", [])
    assert "independentemente de source/exchange" in vazio, vazio
    print("OK  mensagem de 'sem dados' lista o que existe na tabela")


# --------------------------------------------------------------------- M2
def _base_cfg():
    return {
        "framework": "keras", "model_type": "lstm",
        "normalization": {"strategy": "local", "x_mode": "minmax", "y_mode": "minmax_target"},
        "parallel": {"enabled": False},
        "window_size": 96, "steps_ahead": 1,
    }


def test_combos_do_grid_nao_compartilham_dicts_aninhados():
    """M2: cópia rasa deixava combinações mexendo no mesmo dicionário."""
    svc = ModelService()
    base = _base_cfg()

    cfg = svc._apply_params_and_seed_unified(base, {"SEED": 7})
    cfg["normalization"]["x_mode"] = "zscore"
    cfg["parallel"]["enabled"] = True

    assert base["normalization"]["x_mode"] == "minmax", "vazou para o config base"
    assert base["parallel"]["enabled"] is False, "vazou para o config base"
    print("OK  cada combinação recebe cópia profunda da configuração")


# --------------------------------------------------------------------- M3
def test_cada_combo_tem_run_id_proprio():
    """M3: run_id herdado fazia todas as combinações gravarem na mesma pasta."""
    svc = ModelService()
    base = _base_cfg()
    base["run_id"] = "meu_run"

    a = svc._apply_params_and_seed_unified(base, {"SEED": 1})
    b = svc._apply_params_and_seed_unified(base, {"SEED": 2})

    assert a["run_id"] != b["run_id"], "duas combinações com o mesmo run_id"
    assert a["run_id"].startswith("meu_run"), "o prefixo do usuário deve ser preservado"
    print(f"OK  run_id único por combinação: {a['run_id']} != {b['run_id']}")


# --------------------------------------------------------------------- C2 e M5
def test_score_on_aceita_validacao():
    """C2: seleção de hiperparâmetro precisa poder pontuar na validação."""
    ModelService.validar_score("rmse", "validation")
    ModelService.validar_score("r2", "train")
    ModelService.validar_score("mae", "test")
    for score, alvo in [("acuracia", "validation"), ("rmse", "producao")]:
        try:
            ModelService.validar_score(score, alvo)
        except ValueError:
            continue
        raise AssertionError(f"deveria recusar score={score} score_on={alvo}")
    print("OK  score_on aceita validation e recusa valor inválido")


def test_score_ausente_nao_derruba_o_grid():
    """M5: métrica ausente virava TypeError e interrompia o grid inteiro."""
    metricas = {"train": {"rmse": 1.0}, "validation": {"rmse": 2.0}, "test": {}}

    assert ModelService.extrair_score(metricas, "rmse", "validation") == 2.0
    assert ModelService.extrair_score(metricas, "rmse", "test") is None
    assert ModelService.extrair_score(metricas, "mae", "train") is None
    assert ModelService.extrair_score({}, "rmse", "validation") is None
    print("OK  métrica ausente devolve None em vez de explodir")


# --------------------------------------------------------------------- M1
def test_impasse_no_grid_paralelo_e_detectado():
    """M1: nenhum job admitido com nada rodando era espera ocupada infinita."""
    # nada rodando, nada admitido: conta tentativa
    assert ModelService.impasse_no_escalonador(pendentes=3, rodando=0, admitiu=False,
                                               tentativas=1, limite=30) is False
    assert ModelService.impasse_no_escalonador(pendentes=3, rodando=0, admitiu=False,
                                               tentativas=31, limite=30) is True
    # há jobs rodando: nunca é impasse, basta esperar terminarem
    assert ModelService.impasse_no_escalonador(pendentes=3, rodando=2, admitiu=False,
                                               tentativas=99, limite=30) is False
    # admitiu alguém: houve progresso
    assert ModelService.impasse_no_escalonador(pendentes=3, rodando=0, admitiu=True,
                                               tentativas=99, limite=30) is False
    print("OK  impasse do escalonador é detectado em vez de girar em vazio")


# --------------------------------------------------------------------- M6
def test_fabrica_de_trainers_sem_exec_nem_eval():
    """M6: importação dinâmica por exec/eval trocada por importlib."""
    fonte = open(os.path.join(SRC, "services", "trainer_factory.py"), encoding="utf-8").read()
    assert "exec(" not in fonte, "ainda usa exec"
    assert "eval(" not in fonte, "ainda usa eval"
    assert "importlib" in fonte, "deveria importar via importlib"

    cls = TrainerFactory.get_trainer("keras", "lstm")
    assert cls.__name__ == "KerasLSTMTrainer", cls.__name__

    for fw, mt in [("naoexiste", "lstm"), ("keras", "naoexiste")]:
        try:
            TrainerFactory.get_trainer(fw, mt)
        except ValueError:
            continue
        raise AssertionError(f"deveria recusar {fw}/{mt}")
    print("OK  fábrica de trainers usa importlib e valida entrada")


def test_framework_sem_dependencia_explica_o_extra():
    """M9: pytorch e MetaTrader5 nao estao no ambiente publicado."""
    try:
        import torch  # noqa: F401
        print("OK  torch instalado; nada a verificar")
        return
    except ImportError:
        pass

    try:
        TrainerFactory.get_trainer("pytorch", "lstm")
    except ImportError as e:
        assert "requirements-optional.txt" in str(e), str(e)
        print("OK  framework sem dependência aponta o arquivo de extras")
        return
    raise AssertionError("deveria falhar sem torch instalado")


# --------------------------------------------------------------------- B5
def test_postprocess_recusa_timestamps_incompativeis():
    """B5: cortar pelo fim alinhava séries erradas em silêncio."""
    svc = ModelService()
    y_pred = np.arange(10.0).reshape(-1, 1)
    y_true = np.arange(10.0).reshape(-1, 1)
    ts_ok = np.arange(10)

    out = svc.universal_postprocess(y_pred, y_true, ts_ok)
    assert len(out[2]) == 10

    try:
        svc.universal_postprocess(y_pred, y_true, np.arange(13))
    except ValueError:
        print("OK  postprocess recusa timestamps em quantidade diferente")
        return
    raise AssertionError("deveria recusar timestamps incompatíveis")


# --------------------------------------------------------------------- A5
def test_grid_legado_nao_voltou():
    """A5: a classe antiga ignorava window_size e maximizava o erro.

    O que precisa ser garantido nao e a ausencia do caminho
    `optimization/grid_search.py` — ele hoje hospeda a implementacao correta,
    extraida de ModelService — e sim que a classe `GridSearch` legada, com
    aqueles dois defeitos, nao esteja de volta.
    """
    import optimization.grid_search as g

    assert not hasattr(g, "GridSearch"), (
        "a classe GridSearch legada voltou: ignorava window_size e escolhia a pior "
        "configuracao quando a metrica nao se chamava 'loss'"
    )
    assert callable(getattr(g, "run_grid_search", None)), \
        "a implementacao atual (run_grid_search) deveria estar neste modulo"

    # o defeito de ordenacao: escolher pelo erro exige minimizar, nao maximizar
    fonte = open(os.path.join(SRC, "optimization", "grid_search.py"), encoding="utf-8").read()
    assert 'scoring' not in fonte or 'loss' not in fonte.split('scoring')[0][-200:], \
        "reaparece a ordenacao condicionada ao nome da metrica"

    # e a selecao continua ancorada na validacao, nao no teste
    assert 'score_on: str = "validation"' in fonte, \
        "o padrao de score_on saiu da validacao"

    print("OK  grid legado nao voltou; implementacao atual no modulo proprio")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
    print(f"\n{len(fns)} testes passaram.")
