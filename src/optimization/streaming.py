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


def prepare_block(config: dict, df_ctx, processor, *, inverse_kind: str, scaler_dir: Optional[str] = None):
    """
    Janela e normaliza um bloco para um modelo ja treinado. `df_ctx` deve conter
    o bloco precedido de `window_size + steps_ahead − 1` linhas de contexto (as
    colunas de `relevant_columns` e a coluna alvo). Devolve X normalizado, y
    normalizado, y original (n, H), janelas do alvo e o contexto de inversao.
    Suporta global (scaler salvo em `scaler_dir` ou ja carregado no processor) e
    local; evomsn fica fora do escopo deste adaptador.
    """
    from services.model_service import ModelService

    ws = int(config["window_size"])
    H = int(config.get("steps_ahead", 1))
    target = config["target_column"]
    feats = [c for c in config["relevant_columns"] if c not in (target, "timestamp")]
    data = df_ctx[feats + [target]].reset_index(drop=True)
    X_feat, y = processor.create_windows(data, coluna_alvo=target, steps_ahead=H)   # X sem o alvo
    y = np.asarray(y, dtype=float).reshape(len(y), -1)
    tw = ModelService._build_target_windows(data[target].values, ws, H)
    # _norm_cfg so le o config; passar None como self evita instanciar o servico (banco).
    strategy, _scaler_type, x_mode, y_mode, _ = ModelService._norm_cfg(None, config)
    append_target = ModelService._resolve_include_target_channel(
        ModelService.policy_from_config(config), norm_strategy=strategy, y_mode=y_mode)
    if append_target:
        X_feat = np.concatenate([X_feat, tw], axis=2)
    if inverse_kind == "global":
        if scaler_dir:
            processor.load_scaler(scaler_dir)
        Xs, ys = processor.apply_normalization_global(X_feat, y)
        ctx = None
    elif inverse_kind == "local":
        Xs, ys, ctx = processor.apply_normalization_local(
            X_feat, y, x_mode=x_mode, y_mode=y_mode, target_idx=X_feat.shape[2] - 1, target_windows=tw)
    else:
        raise NotImplementedError("prepare_block cobre apenas 'global' e 'local'.")
    return {"Xs": Xs, "ys": np.asarray(ys, dtype=float).reshape(len(y), -1), "y": y, "tw": tw, "ctx": ctx}


def invert_block(y_scaled, prep: dict, processor, *, inverse_kind: str):
    """Inverte uma previsao normalizada do bloco preparado e devolve (y_hat, scale), ambos (n, H)."""
    n = len(prep["y"])
    H = prep["y"].shape[1]
    y_scaled = np.asarray(y_scaled, dtype=float).reshape(n, -1)
    if inverse_kind == "global":
        return np.asarray(processor.inverse_transform_global(y_scaled)).reshape(n, -1), _scale_global(processor, n, H)
    return np.asarray(processor.inverse_transform_local(y_scaled, prep["ctx"])).reshape(n, -1), _scale_local(prep["ctx"], n, H)


def predict_block(model, config: dict, df_ctx, processor, *, inverse_kind: str, scaler_dir: Optional[str] = None):
    """
    Prediz um bloco com um modelo já treinado, sem retreinar (ver `prepare_block`
    para o formato de `df_ctx`). Devolve y, y_hat, y_last (persistência) e a
    escala usada na inversão, todos 1D (primeiro passo do horizonte).
    """
    prep = prepare_block(config, df_ctx, processor, inverse_kind=inverse_kind, scaler_dir=scaler_dir)
    y_hat, scale = invert_block(model.predict(prep["Xs"], verbose=0), prep, processor, inverse_kind=inverse_kind)
    return {"y": prep["y"][:, 0], "y_hat": y_hat[:, 0], "y_last": prep["tw"][:, -1, 0], "scale": scale[:, 0]}


# ---------------------------------------------------------------- simulacao hora a hora
# O laco por blocos acima e o esqueleto testavel das politicas. A simulacao do
# artigo 5 (Algoritmo 1 do texto) decide a cada HORA: a previsao e emitida, o
# residuo padronizado alimenta o CUSUM e, se houver alarme (ou se completou o
# periodo), o fine-tuning acontece naquela hora e as horas seguintes ja usam o
# modelo novo. `simulate_hourly` implementa isso com funcoes injetadas, para que
# a logica seja testada sem banco nem GPU; `run_streaming.py` liga ao Nahas.
STREAM_POLICIES = {
    # nome: (normalizacao, gatilho)  — gatilho em {"none", "periodic", "cusum", "cusum+periodic"}
    "A": ("global", "none"),
    "B": ("global", "periodic"),
    "C": ("global", "cusum"),
    "D": ("anchored", "none"),
    "E": ("anchored", "cusum"),
    "B_anch": ("anchored", "periodic"),        # controle: B sob normalizacao ancorada
    "E_per": ("anchored", "cusum+periodic"),   # controle: E com agenda periodica alem do gatilho
}


def block_bounds(n_hours: int, block: int) -> List[tuple]:
    """[(inicio, fim)) dos blocos de `block` horas; o ultimo pode ser menor."""
    if block < 1:
        raise ValueError("block deve ser >= 1.")
    return [(s, min(s + block, n_hours)) for s in range(0, n_hours, block)]


def finetune_slices(t: int, *, days: int, val_days: int, embargo: int, hours_per_day: int = 24) -> Dict[str, tuple]:
    """
    Indices (em horas, fechados a esquerda e abertos a direita) do fine-tuning
    decidido na hora t: os ultimos `days` dias terminam em t (inclusive); os
    ultimos `val_days` dias sao a fatia de validacao V; `embargo` horas sao
    descartadas do fim do treino para que nenhuma janela de treino alcance V.
    """
    n = days * hours_per_day
    nv = val_days * hours_per_day
    if nv >= n:
        raise ValueError("val_days deve ser menor que days.")
    fim = t + 1
    ini = fim - n                      # pode ser negativo: horas anteriores ao inicio da simulacao
    val_ini = fim - nv
    tr_fim = val_ini - embargo
    if tr_fim <= ini:
        raise ValueError("days*24 deve exceder val_days*24 + embargo.")
    return {"train": (ini, tr_fim), "val": (val_ini, fim)}


def first_alarm(monitor: CusumMonitor, zs) -> Optional[int]:
    """Alimenta o monitor ate o primeiro alarme; devolve o indice local ou None."""
    for i, z in enumerate(np.asarray(zs, dtype=float).reshape(-1)):
        if monitor.update(z):
            return i
    return None


def simulate_hourly(
    trigger: str,
    n_hours: int,
    predict_fn: Callable,
    finetune_fn: Callable,
    initial_state,
    *,
    monitor: Optional[CusumMonitor] = None,
    period: int = 720,
    block: int = 720,
    rollback_tolerance_pct: float = 0.0,
    replay_fn: Optional[Callable] = None,
) -> Dict[str, object]:
    """
    Simulacao hora a hora de uma regra de atualizacao.

    trigger      "none" | "periodic" | "cusum" | "cusum+periodic"
    predict_fn   (state, ini, fim) -> dict com y, y_hat, y_last, scale para as horas [ini, fim)
    finetune_fn  (state, t) -> (novo_state, val_novo, val_atual, segundos): ajusta o modelo
                 com dados ate a hora t e devolve o RMSE normalizado do modelo novo e do
                 atual na fatia V; o novo so entra se val_novo < val_atual*(1+tol/100).
    replay_fn    (state, ini, fim) -> y_hat: usado para medir o erro que o modelo
                 rejeitado teria no bloco seguinte (custo evitado pelo rollback).
    monitor      CusumMonitor calibrado; obrigatorio se o gatilho usa CUSUM; se dado
                 com gatilho "none"/"periodic", roda passivo (alarmes registrados,
                 nenhuma acao), o que serve ao diagnostico de alarmes falsos.

    Devolve, por hora, y, y_hat, y_last, scale, z, s_plus, s_minus, alarm,
    model_version e block; a lista de eventos; e os totais.
    """
    if trigger not in ("none", "periodic", "cusum", "cusum+periodic"):
        raise ValueError("trigger invalido.")
    usa_cusum = "cusum" in trigger
    usa_periodo = "periodic" in trigger
    if usa_cusum and monitor is None:
        raise ValueError("Gatilho CUSUM exige um CusumMonitor calibrado.")
    cols = {k: np.full(n_hours, np.nan) for k in ("y", "y_hat", "y_last", "scale", "z", "s_plus", "s_minus")}
    alarm = np.zeros(n_hours, dtype=bool)
    version = np.zeros(n_hours, dtype=int)
    blk = np.zeros(n_hours, dtype=int)
    for b, (s, e) in enumerate(block_bounds(n_hours, block)):
        blk[s:e] = b
    state = initial_state
    ver = 0
    eventos: List[Dict[str, object]] = []
    pendente_rejeitado = None            # (state rejeitado, hora da rejeicao)
    n_retrain = n_accept = 0
    gpu_s = 0.0
    t = 0
    ultimo_update = -1                    # hora do ultimo fine-tuning (ou -1)
    while t < n_hours:
        # proximo ponto periodico: 720 h depois do ultimo update (ou do inicio)
        prox_periodico = (ultimo_update + period) if usa_periodo else n_hours
        fim = min(n_hours, prox_periodico + 1) if usa_periodo else n_hours
        if fim <= t:
            fim = min(n_hours, t + 1)
        out = predict_fn(state, t, fim)
        y, y_hat, y_last = (np.asarray(out[k], dtype=float).reshape(-1) for k in ("y", "y_hat", "y_last"))
        scale = np.asarray(out["scale"], dtype=float).reshape(-1)
        z = standardized_residuals(y, y_hat, scale)
        n_seg = len(y)
        j_alarm = None
        if monitor is not None:
            sp = np.empty(n_seg)
            sm = np.empty(n_seg)
            for i in range(n_seg):
                disparou = monitor.update(z[i])
                sp[i], sm[i] = monitor.s_plus, monitor.s_minus
                if disparou:
                    alarm[t + i] = True
                    if usa_cusum:
                        j_alarm = i
                        break
        else:
            sp = sm = np.full(n_seg, np.nan)
        # quantas horas deste segmento ficam valendo
        corte = n_seg if j_alarm is None else j_alarm + 1
        sl = slice(t, t + corte)
        cols["y"][sl], cols["y_hat"][sl], cols["y_last"][sl] = y[:corte], y_hat[:corte], y_last[:corte]
        cols["scale"][sl], cols["z"][sl] = scale[:corte], z[:corte]
        cols["s_plus"][sl], cols["s_minus"][sl] = sp[:corte], sm[:corte]
        version[sl] = ver
        t_evento = t + corte - 1
        gatilho = None
        if j_alarm is not None:
            gatilho = "cusum"
            eventos.append({"type": "alarm", "t": int(t_evento)})
        elif usa_periodo and t_evento == prox_periodico and t_evento < n_hours - 1:
            gatilho = "periodic"
        t = t + corte
        if gatilho is None or t >= n_hours:      # sem gatilho, ou alarme na ultima hora
            continue
        novo, val_novo, val_atual, seg = finetune_fn(state, t_evento)
        n_retrain += 1
        gpu_s += float(seg)
        aceito = bool(val_novo < val_atual * (1.0 + rollback_tolerance_pct / 100.0))
        ev = {"type": "finetune", "t": int(t_evento), "trigger": gatilho, "accepted": aceito,
              "val_rmse_new": float(val_novo), "val_rmse_current": float(val_atual),
              "seconds": float(seg), "version_before": ver}
        if aceito:
            state, ver = novo, ver + 1
            n_accept += 1
        else:
            pendente_rejeitado = (novo, t_evento, len(eventos))
        eventos.append(ev)
        ultimo_update = t_evento
        if monitor is not None:
            monitor.s_plus = monitor.s_minus = 0.0
        # replay do modelo rejeitado no bloco seguinte (so diagnostico)
        if pendente_rejeitado is not None and replay_fn is not None:
            st_rej, t_rej, idx_ev = pendente_rejeitado
            fim_rep = min(n_hours, t_rej + 1 + block)
            if fim_rep > t_rej + 1:
                yh_rej = np.asarray(replay_fn(st_rej, t_rej + 1, fim_rep), dtype=float).reshape(-1)
                # o modelo mantido ainda vai prever essas horas; guarda para comparar depois
                eventos[idx_ev]["replay"] = {"from": int(t_rej + 1), "to": int(fim_rep), "y_hat_rejected": yh_rej}
            pendente_rejeitado = None
    # completa o replay com o erro do modelo mantido nas mesmas horas
    for ev in eventos:
        rep = ev.get("replay")
        if rep is None:
            continue
        a, b = rep["from"], rep["to"]
        yy = cols["y"][a:b]
        ok = np.isfinite(yy)
        yh_rej = rep.pop("y_hat_rejected")[: b - a]
        rep["rmse_rejected"] = float(np.sqrt(np.mean((yy[ok] - yh_rej[ok]) ** 2))) if ok.any() else float("nan")
        rep["rmse_kept"] = float(np.sqrt(np.mean((yy[ok] - cols["y_hat"][a:b][ok]) ** 2))) if ok.any() else float("nan")
    return {"hours": {**cols, "alarm": alarm, "model_version": version, "block": blk},
            "events": eventos, "n_retrain": n_retrain, "n_accepted": n_accept,
            "n_rejected": n_retrain - n_accept, "n_alarms": int(alarm.sum()), "gpu_seconds": gpu_s,
            "final_version": ver}
