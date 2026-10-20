# optimization/streaming.py
"""
Simulação de fluxo contínuo (artigo 5): cinco políticas de adaptação de um
previsor sob mudança de regime, avaliadas bloco a bloco (ex.: mês a mês).

  A  sem adaptação
  B  fine-tuning periódico (a cada `period` blocos)
  C  fine-tuning disparado por CUSUM sobre resíduos padronizados
  D  normalização por janela/ancorada sem fine-tuning (a escolha da normalização
     está no config; a política em si é igual a A)
  E  D + CUSUM (igual a C sob normalização por janela)

O laço é genérico: recebe duas funções, `predict_fn(state, block) -> dict` com
y, y_hat, y_last (persistência) e scale, e `finetune_fn(state, upto_block) ->
(new_state, val_score)`. Isso permite testar a lógica sem banco nem GPU e
ligá-la ao `ModelService` pelo adaptador `nahas_block_functions`.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from utils.baselines import regression_metrics
from utils.cusum import CusumMonitor, standardized_residuals

POLICIES = ("A", "B", "C", "D", "E")


def _theil_u(y, y_hat, y_last) -> float:
    m = regression_metrics(y, y_hat)["rmse"]
    p = regression_metrics(y, y_last)["rmse"]
    return float(m / p) if p > 0 else float("nan")


def simulate_policy(
    policy: str,
    blocks: Sequence,
    predict_fn: Callable,
    finetune_fn: Callable,
    initial_state,
    *,
    monitor: Optional[CusumMonitor] = None,
    period: int = 1,
    rollback_tolerance_pct: float = 5.0,
) -> Dict[str, object]:
    """
    Executa uma política sobre a sequência de blocos. Em B, C e E o fine-tuning
    só substitui o modelo corrente se `val_score` (RMSE normalizado na janela de
    validação deslizante, menor é melhor) não piorar mais do que
    `rollback_tolerance_pct` em relação ao modelo corrente; senão há rollback.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy deve ser uma de {POLICIES}.")
    if policy in ("C", "E") and monitor is None:
        raise ValueError("Políticas C e E exigem um CusumMonitor calibrado.")
    state = initial_state
    current_val: Optional[float] = None
    linhas: List[Dict[str, object]] = []
    n_retrain = n_rollback = 0
    for i, block in enumerate(blocks):
        out = predict_fn(state, block)
        y, y_hat, y_last = out["y"], out["y_hat"], out["y_last"]
        z = standardized_residuals(y, y_hat, out.get("scale"))
        alarms = monitor.update_many(z) if monitor is not None else []
        met = regression_metrics(y, y_hat)
        linha = {"block": i, "label": getattr(block, "label", str(block)),
                 "rmse": met["rmse"], "mae": met["mae"], "theil_u": _theil_u(y, y_hat, y_last),
                 "mean_abs_z": float(np.mean(np.abs(z))), "n_alarms": len(alarms),
                 "retrained": False, "rolled_back": False}

        gatilho = False
        if policy == "B" and period > 0 and (i + 1) % period == 0:
            gatilho = True
        if policy in ("C", "E") and alarms:
            gatilho = True
        if gatilho and i < len(blocks) - 1:
            new_state, val_score = finetune_fn(state, block)
            n_retrain += 1
            linha["retrained"] = True
            if current_val is not None and val_score > current_val * (1 + rollback_tolerance_pct / 100.0):
                n_rollback += 1
                linha["rolled_back"] = True
            else:
                state, current_val = new_state, val_score
        linhas.append(linha)
    return {"policy": policy, "blocks": linhas, "n_retrain": n_retrain, "n_rollback": n_rollback,
            "rmse_mean": float(np.mean([r["rmse"] for r in linhas])),
            "theil_u_mean": float(np.nanmean([r["theil_u"] for r in linhas]))}


def simulate_all(blocks, predict_fn, finetune_fn, initial_state, *, monitor_factory=None,
                 period: int = 1, policies: Sequence[str] = POLICIES) -> Dict[str, Dict]:
    """Roda cada política do zero (estado inicial e monitor novos)."""
    out = {}
    for p in policies:
        mon = monitor_factory() if (monitor_factory is not None and p in ("C", "E")) else None
        out[p] = simulate_policy(p, blocks, predict_fn, finetune_fn, initial_state,
                                 monitor=mon, period=period)
    return out


# ---------------------------------------------------------------- adaptador Nahas
# As duas funcoes abaixo espelham utils.conformal.scale_from_context (branch do
# artigo 3) para esta branch nao depender daquela; unificar ao integrar as duas.
def _scale_local(ctx, n: int, H: int) -> np.ndarray:
    y = ctx.get("y", {})
    mode = y.get("y_mode", "none")
    e = float(y.get("eps", 1e-8))
    if mode == "none":
        s = np.ones((n, 1))
    elif mode == "relative_last":
        s = np.abs(np.asarray(y["y_ref_last"], dtype=float)) + e
    else:
        s = np.asarray(y["y_scale"], dtype=float) + e
    s = s.reshape(n, -1)
    return np.broadcast_to(s, (n, H)).copy() if s.shape[1] == 1 else s[:, :H]


def _scale_global(processor, n: int, H: int) -> np.ndarray:
    sc = processor._scaler_y
    base = np.asarray(getattr(sc, "data_range_", getattr(sc, "scale_", None)), dtype=float).reshape(1, -1)
    if base.shape[1] == 1:
        base = np.repeat(base, H, axis=1)
    return np.broadcast_to(base[:, :H], (n, H)).copy()


def predict_block(model, config: dict, df_ctx, processor, *, inverse_kind: str, scaler_dir: Optional[str] = None):
    """
    Prediz um bloco com um modelo já treinado, sem retreinar. `df_ctx` deve
    conter o bloco precedido de `window_size + steps_ahead − 1` linhas de contexto
    (as colunas de `relevant_columns` e a coluna alvo). Suporta as estratégias
    global (scaler salvo em `scaler_dir`) e local (estatísticas por janela);
    evomsn fica fora do escopo deste adaptador.
    """
    from services.model_service import ModelService

    ws = int(config["window_size"])
    H = int(config.get("steps_ahead", 1))
    target = config["target_column"]
    feats = [c for c in config["relevant_columns"] if c not in (target, "timestamp")]
    data = df_ctx[feats + [target]].reset_index(drop=True)
    X_feat, y = processor.create_windows(data, coluna_alvo=target, steps_ahead=H)   # X sem o alvo
    tw = ModelService._build_target_windows(data[target].values, ws, H)
    # _norm_cfg so le o config; passar None como self evita instanciar o servico (banco).
    strategy, _scaler_type, x_mode, y_mode, _ = ModelService._norm_cfg(None, config)
    append_target = ModelService._resolve_include_target_channel(
        ModelService.policy_from_config(config), norm_strategy=strategy, y_mode=y_mode)
    if append_target:
        X_feat = np.concatenate([X_feat, tw], axis=2)
    y_last = tw[:, -1, 0]

    if inverse_kind == "global":
        if scaler_dir:
            processor.load_scaler(scaler_dir)
        Xs, _ = processor.apply_normalization_global(X_feat, y)
        y_hat = processor.inverse_transform_global(model.predict(Xs, verbose=0))
        scale = _scale_global(processor, len(y), H)
    elif inverse_kind == "local":
        Xs, _, ctx = processor.apply_normalization_local(
            X_feat, y, x_mode=x_mode, y_mode=y_mode, target_idx=X_feat.shape[2] - 1, target_windows=tw)
        y_hat = processor.inverse_transform_local(model.predict(Xs, verbose=0), ctx)
        scale = _scale_local(ctx, len(y), H)
    else:
        raise NotImplementedError("predict_block cobre apenas 'global' e 'local'.")
    return {"y": y[:, 0], "y_hat": np.asarray(y_hat)[:, 0], "y_last": y_last, "scale": np.asarray(scale)[:, 0]}
