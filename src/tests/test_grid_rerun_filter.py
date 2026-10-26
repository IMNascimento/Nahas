"""Selecao de combinacoes para reexecucao (run_grid.py --rerun-failed / --rerun-filter)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimization.grid_search import split_rerun  # noqa: E402

PREV = [
    {"run_id": "a1", "params": {"MODEL_TYPE": "lstm", "SEED": 1}},
    {"run_id": "b2", "params": {"MODEL_TYPE": "transformer", "SEED": 1}},
    {"run_id": None, "params": {"MODEL_TYPE": "transformer", "SEED": 2}},
    {"run_id": "c3", "params": {"MODEL_TYPE": "linear", "SEED": 1}},
]


def test_so_falhas():
    mantidos, rerun, sub = split_rerun(PREV, True, None)
    assert [r["run_id"] for r in mantidos] == ["a1", "b2", "c3"]
    assert rerun == [{"MODEL_TYPE": "transformer", "SEED": 2}] and sub == []


def test_filtro_por_backbone_inclui_falhas_e_substitui_concluidas():
    mantidos, rerun, sub = split_rerun(PREV, False, {"MODEL_TYPE": "LSTM"})
    assert [r["run_id"] for r in mantidos] == ["b2", "c3"]
    assert rerun == [{"MODEL_TYPE": "lstm", "SEED": 1}, {"MODEL_TYPE": "transformer", "SEED": 2}]
    assert sub == ["a1"]


def test_filtro_com_duas_chaves():
    _, rerun, sub = split_rerun(PREV, False, {"MODEL_TYPE": "transformer", "SEED": "1"})
    assert sub == ["b2"] and len(rerun) == 2
