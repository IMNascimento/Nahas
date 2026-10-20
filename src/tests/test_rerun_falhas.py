"""
Reexecucao das combinacoes que falharam numa grade (run_grid.py --rerun-failed).

Rodar de dentro de `src/`:
    ./venv/bin/python -m pytest tests/test_rerun_falhas.py -q
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import optimization.grid_search as gs                      # noqa: E402
from services.model_service import ModelService            # noqa: E402


class _ServicoFake(ModelService):
    """Mesmos helpers do ModelService; `train` so registra o que foi chamado."""
    def __init__(self):
        super().__init__()
        self.treinados = []

    def train(self, cfg, fw, mt, artifacts_base=None):
        self.treinados.append({"SYMBOL": cfg["symbol"], "MODEL_TYPE": mt})
        rid = f"novo_{len(self.treinados)}"
        return {"run_id": rid, "metrics": {"validation": {"rmse": 0.5}, "readable": {}},
                "model_path": None, "csv_path": None, "metrics_path": f"{rid}/metrics.json"}


def _unified():
    return {"grid_id": "grade_teste", "framework": "keras", "window_size": 8, "steps_ahead": 1,
            "parallel": {"enabled": False}, "use_gpu": False,
            "normalization": {"strategy": "global", "scaler_type": "standard"},
            "SYMBOL": ["AAA", "BBB"], "MODEL_TYPE": ["lstm", "linear"]}


def test_rerun_so_refaz_as_falhas_e_funde_com_as_concluidas(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(gs, "ensure_db_connection", lambda: (_ for _ in ()).throw(RuntimeError("sem banco")))
    root = tmp_path / "results" / "grid" / "grade_teste"
    root.mkdir(parents=True)
    ok = {"run_id": "antigo_1", "params": {"SYMBOL": "AAA", "MODEL_TYPE": "lstm"}, "framework": "keras",
          "model_type": "lstm", "score_on": "validation", "score_name": "rmse", "score_value": 0.9,
          "model_path": None, "csv_path": None, "metrics_path": "antigo_1/metrics.json", "metrics_readable": {}}
    falhas = [dict(ok, run_id=None, params=p, score_value=float("inf"), metrics_path=None,
                   metrics_readable={"error": "Treino excedeu 2.0 horas"})
              for p in ({"SYMBOL": "BBB", "MODEL_TYPE": "lstm"}, {"SYMBOL": "AAA", "MODEL_TYPE": "linear"})]
    (root / "grid_summary.json").write_text(json.dumps({"grid_id": "grade_teste", "summary": [ok, *falhas]}))
    (root / "grid_provenance.json").write_text(json.dumps({"grid_id": "grade_teste", "started_at": "antes"}))

    assert sorted(f["SYMBOL"] + f["MODEL_TYPE"] for f in gs.combos_falhos(str(root))) == ["AAAlinear", "BBBlstm"]

    svc = _ServicoFake()
    out = gs.run_grid_search(svc, _unified(), score="rmse", score_on="validation", rerun_failed=True)

    # So as duas falhas foram treinadas (nao a concluida nem as que nem estavam no summary)
    assert sorted(t["SYMBOL"] + t["MODEL_TYPE"] for t in svc.treinados) == ["AAAlinear", "BBBlstm"]
    ids = sorted(r["run_id"] for r in out["summary"])
    assert ids == ["antigo_1", "novo_1", "novo_2"]
    assert out["best"]["run_id"].startswith("novo_")            # 0.5 < 0.9

    summary = json.loads((root / "grid_summary.json").read_text())
    assert all(r["run_id"] for r in summary["summary"]) and len(summary["summary"]) == 3
    prov = json.loads((root / "grid_provenance.json").read_text())
    assert prov["rerun_failed"] is True and prov["n_combos"] == 3 and prov["n_ok"] == 3
    assert prov["history"][0]["started_at"] == "antes"
    with open(root / "leaderboard.csv") as f:
        assert len(f.read().strip().splitlines()) == 4


def test_rerun_sem_summary_anterior_e_recusado(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, "BASE_DIR", str(tmp_path))
    try:
        gs.run_grid_search(_ServicoFake(), _unified(), rerun_failed=True)
    except FileNotFoundError:
        return
    raise AssertionError("deveria exigir grid_summary.json")
