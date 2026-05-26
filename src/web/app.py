import sys
import os
import threading
import streamlit as st
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import uuid
from db_helpers import list_all_models
from components.model_dropdown import model_dropdown
from components.gpu_selector import gpu_selector

import datetime
from services.model_service import ModelService
import time
from database.model_base import db

if db.is_closed():
    db.connect()

service = ModelService()
live_run_stop_flag = threading.Event()

# ---------------------------------
# CONSTANTES PARA OPÇÕES DE DADOS
# ---------------------------------
AVAILABLE_SYMBOLS = ["BTCUSDT", "BTCBRL", "ETHUSDT", "ETHBRL"]
AVAILABLE_CURRENCIES = ["USDT", "BRL"]
AVAILABLE_EXCHANGES = ["PUBLIC", "BINANCE"]
AVAILABLE_SOURCES = ["binance", "kaggle"]
AVAILABLE_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
# -- Parâmetros para opções
frameworks = ["keras", "pytorch", "tensorflow"]
model_types = ["lstm", "transformer"]
financial_columns = ["close", "open", "high", "low", "volume"]
indicator_options = [
    {"label": "SMA", "key": "sma", "param_label": "Período", "default": 14},
    {"label": "EMA", "key": "ema", "param_label": "Período", "default": 14},
]

# =========================
# Helpers p/ Grid Unificado
# =========================

def _cast_atom(s):
    """Tenta converter para bool/int/float; mantém 'RANDOM' como string; caso contrário devolve string."""
    if isinstance(s, (int, float, bool)):
        return s
    if not isinstance(s, str):
        return s
    v = s.strip()
    if v.lower() == "true":  return True
    if v.lower() == "false": return False
    if v.upper() == "RANDOM": return "RANDOM"
    try:
        return int(v)
    except:
        pass
    try:
        return float(v)
    except:
        pass
    return v

def _parse_sweep_csv(text: str):
    """
    "72,96" -> [72,96]; "Adam,SGD" -> ["Adam","SGD"]; vazio -> [].
    """
    if text is None:
        return []
    s = str(text).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    parts = [p for p in parts if p != ""]
    return [_cast_atom(p) for p in parts]

def _json_download_bytes(pyobj) -> bytes:
    return json.dumps(pyobj, indent=2).encode("utf-8")

def _detect_gpus():
    """
    Retorna lista de rótulos de GPU, e.g. ["0 - NVIDIA A100", "1 - NVIDIA A100"].
    Tenta PyTorch, depois TensorFlow. Se nada achar, [].
    """
    gpus = []
    # PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                gpus.append(f"{i} - {name}")
    except Exception:
        pass
    # TensorFlow
    if not gpus:
        try:
            import tensorflow as tf
            phys = tf.config.list_physical_devices("GPU")
            for i, dev in enumerate(phys):
                # dev.name costuma vir como '/physical_device:GPU:0'
                gpus.append(f"{i} - {getattr(dev, 'name', str(dev))}")
        except Exception:
            pass
    return gpus

# -----------------------------
# Helpers (UI)
# -----------------------------
def _index_or_default(options, value, default_idx=0):
    try:
        return options.index(value)
    except Exception:
        return default_idx

def normalization_ui(defaults: dict | None = None, *, key_prefix: str):
    """
    UI para normalização (global/local/evomsn/evomsn_like).
    Retorna dict para config["normalization"].
    """
    defaults = defaults or {}
    strategy_default = (defaults.get("strategy") or "global").lower()
    strategy = st.radio(
        "Estratégia de normalização",
        options=["global", "local", "evomsn", "evomsn_like"],
        index=_index_or_default(["global", "local", "evomsn", "evomsn_like"], strategy_default, 0),
        key=f"{key_prefix}_norm_strategy",
        horizontal=True,
    )

    if strategy == "global":
        scaler_default = (defaults.get("scaler_type") or "robust").lower()
        scaler_type = st.selectbox(
            "Scaler (Global)",
            options=["standard", "minmax", "robust"],
            index=_index_or_default(["standard", "minmax", "robust"], scaler_default, 2),
            key=f"{key_prefix}_scaler_type",
            help="Um único scaler treinado no treino e reutilizado em val/teste (persistido)."
        )
        return {"strategy": "global", "scaler_type": scaler_type}

    if strategy == "local":
        x_default = (defaults.get("x_mode") or "zscore").lower()
        y_default = (defaults.get("y_mode") or "none").lower()
        x_mode = st.selectbox(
            "Normalização de X (Local)",
            options=["zscore", "minmax", "robust"],
            index=_index_or_default(["zscore", "minmax", "robust"], x_default, 0),
            key=f"{key_prefix}_x_mode",
            help="Normaliza por janela (amostra-a-amostra)."
        )
        y_mode = st.selectbox(
            "Normalização de y (Local)",
            options=["none", "relative_last", "zscore_target", "minmax_target", "robust_target"],
            index=_index_or_default(["none", "relative_last", "zscore_target", "minmax_target", "robust_target"], y_default, 0),
            key=f"{key_prefix}_y_mode",
            help="Se usar *_target ou relative_last, o alvo da janela é usado como referência (o app adiciona a janela do alvo ao X automaticamente)."
        )
        st.caption("ℹ️ Em modos locais com dependência do alvo, o serviço adiciona a janela do alvo ao X como último canal.")
        return {"strategy": "local", "x_mode": x_mode, "y_mode": y_mode}

    if strategy == "evomsn":
        ev_k = int(defaults.get("evomsn_k_scales", 3))
        ev_pred = (defaults.get("evomsn_predictor") or "linear").lower()
        c1, c2 = st.columns([1,1])
        with c1:
            k_scales = st.number_input(
                "EvoMSN: top-k períodos (FFT)",
                min_value=1, max_value=8, value=ev_k, step=1, key=f"{key_prefix}_ev_k",
                help="Número de escalas (períodos) selecionados via FFT."
            )
        with c2:
            predictor = st.selectbox(
                "EvoMSN: preditor (φ̂, ξ̂)",
                options=["linear", "mlp"],
                index=_index_or_default(["linear", "mlp"], ev_pred, 0),
                key=f"{key_prefix}_ev_pred",
                help="Regressor para prever estatísticas futuras por fatia."
            )
        st.caption("EvoMSN do artigo: fatiamento multi-escala, normalização por fatia e predição (φ̂, ξ̂); agregação por potência espectral (FFT).")
        return {
            "strategy": "evomsn",
            "evomsn_k_scales": int(k_scales),
            "evomsn_predictor": predictor,
        }

    # evomsn_like
    like_alpha = float(defaults.get("evomsn_like_alpha", 0.1))
    like_beta  = float(defaults.get("evomsn_like_beta", 0.1))
    like_eps   = float(defaults.get("evomsn_like_eps", 1e-8))
    like_noise = float(defaults.get("evomsn_like_noise_std", 0.0))
    c1, c2 = st.columns([1,1])
    with c1:
        alpha = st.number_input("EvoMSN-like: α (EMA centro)", min_value=0.0, max_value=1.0, value=like_alpha, step=0.01, key=f"{key_prefix}_like_alpha")
        eps   = st.number_input("EvoMSN-like: ε (evitar div/0)", value=like_eps, format="%.1e", key=f"{key_prefix}_like_eps")
    with c2:
        beta  = st.number_input("EvoMSN-like: β (EWM-STD escala)", min_value=0.0, max_value=1.0, value=like_beta, step=0.01, key=f"{key_prefix}_like_beta")
        noise = st.number_input("Ruído gaussiano (std) opcional", min_value=0.0, value=like_noise, step=0.01, key=f"{key_prefix}_like_noise")
    st.caption("EvoMSN-like: por janela, remove tendência via EMA(α) e escala via EWM-STD(β). y é normalizado usando EMA/STD **do alvo** na borda da janela. Ruído opcional no X.")
    return {
        "strategy": "evomsn_like",
        "evomsn_like_alpha": float(alpha),
        "evomsn_like_beta": float(beta),
        "evomsn_like_eps": float(eps),
        "evomsn_like_noise_std": float(noise),
    }

# ---------------------------------
# Config da página
# ---------------------------------
st.set_page_config(
    page_title="Nahas - Plataforma de Modelos Financeiros",
    layout="wide",
    page_icon=":chart_with_upwards_trend:",
)

# --- Session State: inicialização robusta
_ss_defaults = {
    "selected_financial": list(financial_columns),
    "indicators_apply": {},
    "indicator_columns": [],
    "relevant_columns": list(financial_columns),
    "live_run_active": False,
    "last_live_run_result": None,
}
for k, v in _ss_defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def update_feature_list():
    st.session_state.indicator_columns = []
    st.session_state.indicators_apply = {}
    for ind in indicator_options:
        checked = st.session_state.get(f"checked_{ind['key']}", False)
        if checked:
            period = st.session_state.get(f"param_{ind['key']}", ind["default"])
            col_name = f"{ind['key']}_{period}"
            st.session_state.indicator_columns.append(col_name)
            st.session_state.indicators_apply[ind["key"]] = [{"period": period, "col_name": col_name}]
    all_cols = st.session_state.selected_financial + st.session_state.indicator_columns
    st.session_state.all_possible_columns = all_cols
    st.session_state.relevant_columns = [col for col in st.session_state.relevant_columns if col in all_cols]

with st.sidebar:
    st.title("Nahas ML Platform")
    st.markdown("Powered by SophiaMind")
    st.markdown("### Acesso Rápido")
    st.markdown("- [Documentação](../docs/README.md)")
    st.markdown("- [Contato/Suporte](mailto:contato@sophialabs.com.br)")

tabs = st.tabs([
    "Treino Normal",
    "Grid Search",
    "Fine-tuning",
    "Live Run"
])

# ---------------------------------
# 1. TREINO NORMAL
# ---------------------------------
with tabs[0]:
    st.header("Treinamento de Modelo Único")
    st.info("Defina e treine um modelo único com os hiperparâmetros desejados.")

    st.subheader("Escolha do Modelo e o Framework")
    framework = st.selectbox("Framework", frameworks, key="framework_train")
    model_type = st.selectbox("Modelo", model_types, key="model_type_train")
    use_gpu, gpu_index, gpu_label = gpu_selector(framework, key_prefix="train")

    with st.expander("💾 Fonte e Características dos Dados", expanded=True):
        st.markdown("**Defina qual ativo e período você deseja usar para treinar o modelo**")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            symbol = st.selectbox(
                "Symbol (Símbolo do Ativo)", 
                options=AVAILABLE_SYMBOLS,
                index=0,  # default: BTCUSDT
                key="train_symbol",
                help="Escolha o par de negociação"
            )
        with col2:
            interval = st.selectbox(
                "Interval (Intervalo Temporal)",
                options=AVAILABLE_INTERVALS,
                index=3,  # default: 1h
                key="train_interval",
                help="Intervalo dos candles"
            )
        with col3:
            currency = st.selectbox(
                "Currency (Moeda de Cotação)",
                options=AVAILABLE_CURRENCIES,
                index=0,  # default: USDT
                key="train_currency",
                help="Moeda de cotação do ativo"
            )
        
        col4, col5 = st.columns(2)
        with col4:
            exchange = st.selectbox(
                "Exchange",
                options=["Todas"] + AVAILABLE_EXCHANGES,  # Opção "Todas" = None
                index=0,  # default: Todas
                key="train_exchange",
                help="Exchange de origem dos dados"
            )
        with col5:
            source = st.selectbox(
                "Source (Fonte dos Dados)",
                options=["Todas"] + AVAILABLE_SOURCES,
                index=0,  # default: Todas
                key="train_source",
                help="Fonte dos dados históricos"
            )
        
        exchange_value = None if exchange == "Todas" else exchange
        source_value = None if source == "Todas" else source
        
        st.caption(
            f"🔍 Modelo será identificado como: **{symbol}_{interval}_{currency}**"
            f"{f'_{exchange.upper()[:3]}' if exchange else ''}"
        )

    st.subheader("Colunas Financeiras e alvos (target)")
    indicator_columns = []
    indicators_apply = {}

    with st.expander("Indicadores Técnicos"):
        if st.checkbox("SMA", key="ind_sma"):
            sma_period = st.number_input("SMA - Período", value=14, key="sma_period")
            sma_col_name = f"sma_{sma_period}"
            indicators_apply["sma"] = [{"period": sma_period, "col_name": sma_col_name}]
            indicator_columns.append(sma_col_name)
        if st.checkbox("EMA", key="ind_ema"):
            ema_period = st.number_input("EMA - Período", value=14, key="ema_period")
            ema_col_name = f"ema_{ema_period}"
            indicators_apply["ema"] = [{"period": ema_period, "col_name": ema_col_name}]
            indicator_columns.append(ema_col_name)

    financial_columns = ["close", "open", "high", "low", "volume"]
    all_possible_columns = financial_columns + indicator_columns

    target_column = st.selectbox(
        "Coluna alvo (target column)",
        options=all_possible_columns,
        index=all_possible_columns.index("close") if "close" in all_possible_columns else 0
    )
    relevant_columns = st.multiselect(
        "Colunas finais usadas no modelo (features)",
        options=all_possible_columns,
        default=all_possible_columns
    )
    use_seed = st.checkbox("Usar seed fixa para reprodução", value=False)

    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # NORMALIZAÇÃO FORA DO FORM (para atualizar instantaneamente)
    st.markdown("### Normalização")
    norm_cfg = normalization_ui(defaults=None, key_prefix="train")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    with st.form("train_form"):
        st.subheader("Hiperparâmetros")

        if use_seed:
            seed = st.number_input("Seed (reprodutibilidade)", min_value=0, max_value=2**32-1, value=42, step=1)
        else:
            seed = None

        start_date = st.date_input("Data Inicial dos dados", value=datetime.date(2017, 8, 18))
        end_date = st.date_input("Data Final dos dados", value=datetime.date(2025, 1, 19))
        st.caption(f"Selecionado: {start_date.strftime('%d/%m/%Y')} até {end_date.strftime('%d/%m/%Y')}")
        train_size = st.number_input("Train Size (fração para treino)", value=0.7, min_value=0.01, max_value=0.99, step=0.01, format="%.2f")
        validation_split = st.number_input("Validation Split (fração para validação)", value=0.15, min_value=0.01, max_value=0.99, step=0.01, format="%.2f")
        window_size = st.number_input("Window Size", value=96)
        batch_size = st.number_input("Batch Size", value=32)
        epochs = st.number_input("Epochs", value=100)
        patience = st.number_input("Patience", value=5)
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.5f")
        dropout = st.number_input("Dropout", value=0.2, format="%.2f")
        optimizer = st.selectbox("Optimizer", ["Adam", "RMSprop", "SGD"])
        steps_ahead = st.number_input("Steps Ahead (nº de passos à frente / saídas)", value=1, min_value=1, max_value=50)
        loss_function = st.selectbox("Função de Perda", ["mean_squared_error", "mean_absolute_error", "mse"])

        if model_type == "lstm":
            layers = st.text_input("Layers (ex: 128,64)", value="128,64")
            bidirectional = st.checkbox("Bidirecional", value=False)
            l1 = st.number_input("L1 Regularization", value=0.0)
            l2 = st.number_input("L2 Regularization", value=0.0)
            activation = st.text_input("Função de Ativação (ex: tanh,relu)", value="tanh,tanh")
            recurrent_dropout = st.number_input("Recurrent Dropout", value=0.0, format="%.2f")
        else:
            num_layers = st.number_input("Num Layers", value=2)
            embed_dim = st.number_input("Embed Dim", value=32)
            num_heads = st.number_input("Num Heads", value=2)
            ff_dim = st.number_input("FF Dim", value=64)
            activation = st.text_input("Função de Ativação (ex: relu,gelu)", value="relu")
            l1 = st.number_input("L1 Regularization", value=0.0)
            l2 = st.number_input("L2 Regularization", value=0.0)

        submit_btn = st.form_submit_button("Treinar")
        if submit_btn:
            config = {
                "window_size": int(window_size),
                "batch_size": int(batch_size),
                "epochs": int(epochs),
                "patience": int(patience),
                "learning_rate": float(learning_rate),
                "dropout": float(dropout),
                "optimizer": optimizer,
                "loss_fn": loss_function,
                "start_date": start_date.strftime("%Y-%m-%d 00:00:00"),
                "end_date": end_date.strftime("%Y-%m-%d 23:59:59"),
                "relevant_columns": list(relevant_columns),
                "target_column": target_column,
                "indicators_apply": indicators_apply,
                "train_size": float(train_size),
                "validation_split": float(validation_split),
                "steps_ahead": int(steps_ahead),
                "output_units": int(steps_ahead),
                "run_id": str(uuid.uuid4())[:8],
                # usa a NORMALIZAÇÃO escolhidinha fora do form
                "normalization": norm_cfg,
                "symbol": symbol,
                "interval": interval,
                "currency": currency,
            }

            if exchange_value:
                config["exchange"] = exchange_value
            if source_value:
                config["source"] = source_value

            if model_type == "lstm":
                config["layers_config"] = [int(x) for x in layers.split(",") if x.strip()]
                config["bidirectional"] = bool(bidirectional)
                config["l1_reg"] = float(l1)
                config["l2_reg"] = float(l2)
                config["activation_functions"] = [x.strip() for x in activation.split(",")]
                config["recurrent_dropout"] = float(recurrent_dropout)
            else:
                config["num_layers"] = int(num_layers)
                config["embed_dim"] = int(embed_dim)
                config["num_heads"] = int(num_heads)
                config["ff_dim"] = int(ff_dim)
                config["activation"] = activation
                config["l1_reg"] = float(l1)
                config["l2_reg"] = float(l2)

            if use_gpu:
                config["use_gpu"] = True
                config["gpu_index"] = gpu_index
            else:
                config["use_gpu"] = False
                config["gpu_index"] = None
            if use_seed:
                config["seed"] = int(seed)

            with st.spinner("Treinando modelo..."):
                result = service.train(config, framework, model_type)
            st.success(f"Modelo treinado! Caminho: {result['model_path']}")
            st.code(json.dumps(result, indent=2))

# ---------------------------------
# 2. GRID SEARCH
# ---------------------------------
with tabs[1]:
    st.header("Grid Search (qualquer campo pode ser fixo ou varrido)")
    st.caption("Use **JSON unificado** (um arquivo só) ou monte tudo pela interface.")

    # ==== Exemplo unificado p/ download ====
    ex_unified = service.example_unified_grid_json()
    st.download_button(
        "📥 Baixar exemplo (JSON unificado)",
        data=_json_download_bytes(ex_unified),
        file_name="grid_unificado_exemplo.json",
        mime="application/json",
        use_container_width=True
    )

    st.divider()

    modo = st.radio(
        "Como você quer configurar?",
        options=["JSON unificado (upload)", "Configurar via interface"],
        horizontal=True
    )

    score = st.selectbox("Métrica de ranking", ["rmse", "mae", "mse", "r2"], index=0)
    score_on = st.selectbox("Avaliar métrica em", ["test", "train"], index=0)

    # ======================================================
    # MODO 1: Upload de JSON UNIFICADO
    # ======================================================
    if modo == "JSON unificado (upload)":
        st.subheader("Upload de JSON Unificado")
        up = st.file_uploader("Envie o arquivo .json com a configuração unificada", type=["json"], key="unified_json")
        unified_cfg = None
        if up is not None:
            try:
                unified_cfg = json.load(up)
                st.success("JSON carregado com sucesso. Prévia abaixo:")
                st.json(unified_cfg)
            except Exception as e:
                st.error(f"Falha ao ler JSON: {e}")

        if st.button("🚀 Executar Grid (JSON unificado)", type="primary", use_container_width=True):
            if not unified_cfg:
                st.error("Envie um JSON unificado válido.")
            else:
                with st.spinner("Executando Grid Search (unificado)..."):
                    res = service.grid_search_unified(unified_cfg, score=score, score_on=score_on)
                st.success(f"Grid finalizado! ID: {res['grid_id']}")
                st.write("🏆 Melhor configuração:")
                st.json(res["best"])
                st.write("📁 Arquivos gerados:")
                st.code(f"- Leaderboard: {res['leaderboard_csv']}\n- Resumo: {res['summary_json']}")

    # ======================================================
    # MODO 2: UI para montar JSON UNIFICADO
    # ======================================================
    if modo == "Configurar via interface":
        st.subheader("Montar Config Unificada pela UI")

        with st.expander("Identificação e Modelo", expanded=True):
            grid_id_ui = st.text_input("grid_id (opcional)", value="grid_ui")
            fw_list = st.multiselect("Framework(s)", frameworks, default=["keras"])
            mt_list = st.multiselect("Model type(s)", model_types, default=["lstm"])
        
        with st.expander("💾 Dados do Ativo (Symbol, Interval, Currency)", expanded=True):
            st.markdown("**Defina os ativos e períodos para varrer no Grid Search**")
            
            symbol_list = st.multiselect(
                "SYMBOL (selecione um ou mais)",
                options=AVAILABLE_SYMBOLS,
                default=["BTCUSDT"],
                key="grid_symbol",
                help="Escolha os pares de negociação para varrer"
            )
            
            interval_list = st.multiselect(
                "INTERVAL (selecione um ou mais)",
                options=AVAILABLE_INTERVALS,
                default=["1h"],
                key="grid_interval",
                help="Escolha os intervalos temporais"
            )
            
            currency_list = st.multiselect(
                "CURRENCY (selecione uma ou mais)",
                options=AVAILABLE_CURRENCIES,
                default=["USDT"],
                key="grid_currency",
                help="Escolha as moedas de cotação"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                exchange_list_ui = st.multiselect(
                    "EXCHANGE (opcional)",
                    options=AVAILABLE_EXCHANGES,
                    default=[],
                    key="grid_exchange",
                    help="Deixe vazio para buscar de todas"
                )
                exchange_list = exchange_list_ui if exchange_list_ui else [None]
            
            with col2:
                source_list_ui = st.multiselect(
                    "SOURCE (opcional)",
                    options=AVAILABLE_SOURCES,
                    default=[],
                    key="grid_source",
                    help="Deixe vazio para buscar de todas"
                )
                source_list = source_list_ui if source_list_ui else [None]
            
            st.caption(
                f"🔍 Grid varrará: {len(symbol_list)} symbol(s) × "
                f"{len(interval_list)} interval(s) × {len(currency_list)} currency(s) = "
                f"**{len(symbol_list) * len(interval_list) * len(currency_list)} combinações de dados**"
            )

        with st.expander("Janela temporal & Split", expanded=True):
            sd = st.text_input("START_DATE (YYYY-MM-DD HH:MM:SS) — pode listar separado por vírgula", "2017-08-18 00:00:00")
            ed = st.text_input("END_DATE (YYYY-MM-DD HH:MM:SS)", "2025-01-19 23:59:59")
            train_list = _parse_sweep_csv(st.text_input("TRAIN_SIZE (ex: 0.7 ou 0.6,0.7)", "0.7"))
            val_list   = _parse_sweep_csv(st.text_input("VALIDATION_SPLIT (ex: 0.15 ou 0.1,0.2)", "0.15"))
            steps_list = _parse_sweep_csv(st.text_input("STEPS_AHEAD (ex: 1 ou 1,3,5)", "1"))

        # =========================
        # Features / Alvo  (NÃO aninhar nada aqui dentro)
        # =========================
        with st.expander("Features / Alvo", expanded=True):
            fin_cols = ["close", "open", "high", "low", "volume"]

            # Conjuntos de features (A e opcional B)
            setA = st.multiselect("Feature set A", fin_cols, default=fin_cols, key="feat_setA")

            setB_enable = st.checkbox("Adicionar Feature set B?", value=False, key="feat_setB_enable")
            setB = st.multiselect("Feature set B", fin_cols, default=["close", "volume"], key="feat_setB") if setB_enable else None

            # Target pode ser lista (para varrer)
            target_list = _parse_sweep_csv(
                st.text_input("TARGET_COLUMN (pode listar múltiplos separados por vírgula)", "close", key="target_csv")
            )

        # =========================
        # Indicadores técnicos (opcional)  (EXPANDER IRMÃO)
        # =========================
        with st.expander("Indicadores técnicos (opcional)", expanded=False):
            ind_cfg = {}  # se ficar vazio => sem indicadores

            col_i1, col_i2 = st.columns(2)
            with col_i1:
                use_sma = st.checkbox("Usar SMA", key="use_sma")
                if use_sma:
                    sma_period = st.number_input("SMA - Período", min_value=1, value=14, step=1, key="sma_period")
                    ind_cfg["sma"] = [{"period": int(sma_period), "col_name": f"sma_{int(sma_period)}"}]

                use_rsi = st.checkbox("Usar RSI", key="use_rsi")
                if use_rsi:
                    rsi_period = st.number_input("RSI - Período", min_value=2, value=14, step=1, key="rsi_period")
                    ind_cfg["rsi"] = [{"period": int(rsi_period), "col_name": f"rsi_{int(rsi_period)}"}]

            with col_i2:
                use_ema = st.checkbox("Usar EMA", key="use_ema")
                if use_ema:
                    ema_period = st.number_input("EMA - Período", min_value=1, value=14, step=1, key="ema_period")
                    ind_cfg["ema"] = [{"period": int(ema_period), "col_name": f"ema_{int(ema_period)}"}]

                use_macd = st.checkbox("Usar MACD", key="use_macd")
                if use_macd:
                    fast = st.number_input("MACD - Fast", min_value=1, value=12, step=1, key="macd_fast")
                    slow = st.number_input("MACD - Slow", min_value=2, value=26, step=1, key="macd_slow")
                    sig  = st.number_input("MACD - Signal", min_value=1, value=9, step=1, key="macd_signal")
                    ind_cfg["macd"] = [{
                        "fast": int(fast), "slow": int(slow), "signal": int(sig),
                        "col_name_macd": f"macd_{int(fast)}_{int(slow)}",
                        "col_name_signal": f"macd_signal_{int(sig)}",
                        "col_name_hist": f"macd_hist_{int(fast)}_{int(slow)}_{int(sig)}"
                    }]

            st.caption("Se nenhum indicador for marcado, será considerado **sem indicadores**.")

            # Varredura com/sem indicadores no grid unificado
            varrer_inds = st.checkbox("Varrer com/sem indicadores no Grid?", value=False, key="inds_sweep_toggle")
            indicators_sweep = ([{}, ind_cfg] if ind_cfg else [{}]) if varrer_inds else (ind_cfg if ind_cfg else {})

        # =========================
        # Normalização (varrer estratégias diferentes)  (EXPANDER IRMÃO)
        # =========================
        with st.expander("Normalização (varrer estratégias diferentes)", expanded=True):
            norm_opts = st.multiselect(
                "Escolha estratégias para varrer",
                ["global", "local", "evomsn", "evomsn_like"],
                default=["global","local","evomsn"]
            )
            norm_list = []
            if "global" in norm_opts:
                scaler = st.selectbox("Global: scaler_type", ["robust","standard","minmax"], index=0, key="norm_global_scaler")
                norm_list.append({"strategy": "global", "scaler_type": scaler})
            if "local" in norm_opts:
                c1, c2 = st.columns(2)
                with c1:
                    xmode = st.selectbox("Local: x_mode", ["zscore","minmax","robust"], index=0, key="norm_local_x")
                with c2:
                    ymode = st.selectbox("Local: y_mode", ["none","relative_last","zscore_target","minmax_target","robust_target"], index=0, key="norm_local_y")
                norm_list.append({"strategy": "local", "x_mode": xmode, "y_mode": ymode})
            if "evomsn" in norm_opts:
                c1, c2 = st.columns(2)
                with c1:
                    k_scales = st.number_input("EvoMSN: k_scales", min_value=1, max_value=8, value=4, step=1, key="norm_ev_k")
                with c2:
                    predictor = st.selectbox("EvoMSN: predictor", ["linear","mlp"], index=0, key="norm_ev_pred")
                norm_list.append({"strategy": "evomsn", "evomsn_k_scales": int(k_scales), "evomsn_predictor": predictor})
            if "evomsn_like" in norm_opts:
                alpha = st.number_input("EvoMSN-like: alpha", min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="norm_like_alpha")
                beta  = st.number_input("EvoMSN-like: beta",  min_value=0.0, max_value=1.0, value=0.1, step=0.01, key="norm_like_beta")
                eps   = st.number_input("EvoMSN-like: eps",   value=1e-8, format="%.1e", key="norm_like_eps")
                noise = st.number_input("EvoMSN-like: noise std", min_value=0.0, value=0.0, step=0.01, key="norm_like_noise")
                norm_list.append({
                    "strategy": "evomsn_like",
                    "evomsn_like_alpha": float(alpha),
                    "evomsn_like_beta": float(beta),
                    "evomsn_like_eps": float(eps),
                    "evomsn_like_noise_std": float(noise),
                })
        
        with st.expander("Execução paralela (GPU/CPU) – opcional", expanded=False):
            parallel_enabled = st.checkbox("Ativar execução paralela com controle de capacidade", value=False, key="par_enabled")
            backend = st.selectbox("Backend", ["process"], index=0, key="par_backend", help="Use 'process' (recomendado).")
            max_workers_per_gpu = st.number_input("Máx. jobs simultâneos por GPU", min_value=1, value=2, step=1, key="par_mwpg")
            safety_ratio = st.number_input("Margem de segurança de memória (ex.: 0.20 = 20%)", min_value=0.0, max_value=0.9, value=0.20, step=0.05, key="par_safety")
            cpu_workers = st.number_input("Máx. jobs simultâneos na CPU", min_value=0, value=2, step=1, key="par_cpu")

        with st.expander("Device & Seed", expanded=False):
            use_gpu_ui = st.checkbox("USE_GPU", value=False, key="grid_use_gpu")

            selected_gpus = []
            if use_gpu_ui:
                gpu_list = _detect_gpus()
                if not gpu_list:
                    st.warning("Nenhuma GPU detectada neste ambiente.")
                else:
                    st.write("Selecione as GPUs que deseja usar:")
                    # se houver só 1 GPU, pré-seleciona
                    default_sel = gpu_list if len(gpu_list) == 1 else st.session_state.get("grid_selected_gpus", [])
                    current_sel = []
                    for label in gpu_list:
                        checked = st.checkbox(label, value=(label in default_sel), key=f"grid_gpu_{label}")
                        if checked:
                            current_sel.append(label)
                    selected_gpus = current_sel
                    st.session_state["grid_selected_gpus"] = selected_gpus

            seed_csv = st.text_input('SEED (ex: "RANDOM" ou 42,123)', value="RANDOM", key="grid_seed_csv")
            seed_list = _parse_sweep_csv(seed_csv) if seed_csv.strip() else []

        with st.expander("Hiperparâmetros (coloque listas para varrer)", expanded=True):
            w_sizes = _parse_sweep_csv(st.text_input("WINDOW_SIZE", "72,96"))
            batches = _parse_sweep_csv(st.text_input("BATCH_SIZE", "16,32"))
            epochs  = _parse_sweep_csv(st.text_input("EPOCHS", "50"))
            pat     = _parse_sweep_csv(st.text_input("PATIENCE", "5,10"))
            lrs     = _parse_sweep_csv(st.text_input("LEARNING_RATE", "0.001,0.0005"))
            drps    = _parse_sweep_csv(st.text_input("DROPOUT", "0.2,0.3"))
            optims  = _parse_sweep_csv(st.text_input("OPTIMIZER", "Adam"))
            losses  = _parse_sweep_csv(st.text_input("LOSS_FUNCTION", "mean_squared_error"))

            lstm_sets, bidirs, act_sets, rec_dp = [], [], [], []
            if "lstm" in mt_list:
                lstm_layers = st.text_input('LAYERS_CONFIG (ex: "128,64" ou várias: "128,64 | 64,64")', "128,64 | 64,64")
                for chunk in [c.strip() for c in lstm_layers.split("|") if c.strip()]:
                    lstm_sets.append([int(x) for x in chunk.split(",") if x.strip()])
                bidirs = _parse_sweep_csv(st.text_input("BIDIRECTIONAL (true,false)", "false,true"))
                acts   = st.text_input('ACTIVATION_FUNCTION (por camada; use "|" p/ múltiplos sets)', "tanh,tanh | relu, relu")
                for chunk in [c.strip() for c in acts.split("|") if c.strip()]:
                    act_sets.append([x.strip() for x in chunk.split(",") if x.strip()])
                rec_dp = _parse_sweep_csv(st.text_input("RECURRENT_DROPOUT", "0.0"))

            n_layers, embedd, heads, ff_dim, act_tr = [], [], [], [], []
            if "transformer" in mt_list:
                n_layers = _parse_sweep_csv(st.text_input("NUM_LAYERS", "2,3"))
                embedd   = _parse_sweep_csv(st.text_input("EMBED_DIM", "32"))
                heads    = _parse_sweep_csv(st.text_input("NUM_HEADS", "2,4"))
                ff_dim   = _parse_sweep_csv(st.text_input("FF_DIM", "64,128"))
                act_tr   = _parse_sweep_csv(st.text_input("ACTIVATION", "relu"))

        # ----- Monta o JSON unificado -----
        unified_cfg = {
            "grid_id": grid_id_ui,
            "framework": fw_list,
            "model_type": mt_list,
            "SYMBOL": symbol_list if symbol_list else ["BTCUSDT"],
            "INTERVAL": interval_list if interval_list else ["1h"],
            "CURRENCY": currency_list if currency_list else ["USDT"],
            "EXCHANGE": exchange_list,
            "SOURCE": source_list,
            "START_DATE": [s.strip() for s in sd.split(",") if s.strip()],
            "END_DATE":   [s.strip() for s in ed.split(",") if s.strip()],
            "TRAIN_SIZE": train_list if train_list else [0.7],
            "VALIDATION_SPLIT": val_list if val_list else [0.15],
            "STEPS_AHEAD": steps_list if steps_list else [1],
            "TARGET_COLUMN": target_list if target_list else ["close"],
            "RELEVANT_COLUMNS": [setA] + ([setB] if setB else []),
            "INDICATORS_APPLY": indicators_sweep,  # dict (fixo) ou lista de dicts
            "NORMALIZATION": norm_list,
            "USE_GPU": [bool(use_gpu_ui and selected_gpus)],  # verdadeiro só se marcou e selecionou algo
            "GPU_INDEX": (selected_gpus if (use_gpu_ui and selected_gpus) else [None]),
            "SEED": seed_list if seed_list else ["RANDOM"],
            "WINDOW_SIZE": w_sizes if w_sizes else [96],
            "BATCH_SIZE": batches if batches else [32],
            "EPOCHS": epochs if epochs else [50],
            "PATIENCE": pat if pat else [5],
            "LEARNING_RATE": lrs if lrs else [0.001],
            "DROPOUT": drps if drps else [0.2],
            "OPTIMIZER": optims if optims else ["Adam"],
            "LOSS_FUNCTION": losses if losses else ["mean_squared_error"],
        }
        if "lstm" in mt_list:
            unified_cfg["LAYERS_CONFIG"] = lstm_sets if lstm_sets else [[128,64]]
            unified_cfg["BIDIRECTIONAL"] = bidirs if bidirs else [False]
            unified_cfg["ACTIVATION_FUNCTION"] = act_sets if act_sets else [["tanh","tanh"]]
            unified_cfg["RECURRENT_DROPOUT"] = rec_dp if rec_dp else [0.0]
        if "transformer" in mt_list:
            unified_cfg["NUM_LAYERS"] = n_layers if n_layers else [2]
            unified_cfg["EMBED_DIM"]  = embedd if embedd else [32]
            unified_cfg["NUM_HEADS"]  = heads if heads else [2]
            unified_cfg["FF_DIM"]     = ff_dim if ff_dim else [64]
            unified_cfg["ACTIVATION"] = act_tr if act_tr else ["relu"]
        
        if parallel_enabled:
            unified_cfg["PARALLEL"] = {
                "enabled": bool(parallel_enabled),
                "backend": backend,
                "max_workers_per_gpu": int(max_workers_per_gpu),
                "safety_ratio": float(safety_ratio),
                "cpu_workers": int(cpu_workers),
            }

        st.markdown("Prévia do JSON unificado que será usado:")
        st.code(json.dumps(unified_cfg, indent=2), language="json")
        st.download_button(
            "💾 Baixar este JSON montado",
            data=_json_download_bytes(unified_cfg),
            file_name="grid_unificado_montado.json",
            mime="application/json",
            use_container_width=True
        )

        if st.button("🚀 Executar Grid (UI → unificado)", type="primary", use_container_width=True):
            if not fw_list or not mt_list:
                st.error("Selecione ao menos 1 framework e 1 model_type.")
            else:
                with st.spinner("Executando Grid Search (unificado)..."):
                    res = service.grid_search_unified(unified_cfg, score=score, score_on=score_on)
                st.success(f"Grid finalizado! ID: {res['grid_id']}")
                st.write("🏆 Melhor configuração:")

                def _to_jsonable(x):
                    import numpy as np
                    if isinstance(x, dict):
                        return {k: _to_jsonable(v) for k, v in x.items()}
                    if isinstance(x, (list, tuple)):
                        return [_to_jsonable(v) for v in x]
                    if isinstance(x, np.generic):  # np.int64, np.float64 etc.
                        return x.item()
                    return x

                best_obj = res.get("best")
                if best_obj:
                    st.json(_to_jsonable(best_obj))
                else:
                    st.warning("Nenhuma execução válida retornou métricas. Veja o leaderboard para detalhes de erros.")

                st.write("📁 Arquivos gerados:")
                st.code(f"- Leaderboard: {res['leaderboard_csv']}\n- Resumo: {res['summary_json']}")




# ---------------------------------
# 3. FINE-TUNING
# ---------------------------------
with tabs[2]:
    st.header("Fine-tuning de Modelo Production-Ready")
    st.info("🔧 Fine-tuning com validação automática, backup e comparação de performance")
    
    models = list_all_models()
    selected_model = model_dropdown(models, key="finetune_model")
    
    if selected_model is not None:
        use_gpu_ft, gpu_index_ft, gpu_label_ft = gpu_selector(selected_model["framework"], key_prefix="finetune")
        
        # === INFORMAÇÕES DO MODELO ORIGINAL ===
        with st.expander("📊 Informações do Modelo Original", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Framework", selected_model["framework"])
                st.metric("Tipo", selected_model["model_type"])
            with col2:
                st.metric("Run ID", selected_model["run_uuid"])
                st.metric("Seed", selected_model.get("seed", "N/A"))
            with col3:
                st.metric("Train Loss", f"{selected_model.get('train_loss', 0):.4f}" if selected_model.get('train_loss') else "N/A")
                st.metric("Val Loss", f"{selected_model.get('val_loss', 0):.4f}" if selected_model.get('val_loss') else "N/A")
            
            st.code(f"Path: {selected_model['model_path']}", language="text")
            
            # Métricas originais se disponíveis
            if selected_model.get("metrics_json"):
                try:
                    orig_metrics = json.loads(selected_model["metrics_json"])
                    st.markdown("**Métricas do Teste Original:**")
                    test_m = orig_metrics.get("test", {})
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("RMSE", f"{test_m.get('rmse', 0):.4f}")
                    with col2:
                        st.metric("MAE", f"{test_m.get('mae', 0):.4f}")
                    with col3:
                        st.metric("MSE", f"{test_m.get('mse', 0):.4f}")
                    with col4:
                        st.metric("R²", f"{test_m.get('r2', 0):.4f}")
                except Exception:
                    pass
        
        # === CONFIGURAÇÃO BASE ===
        with open(selected_model["config_path"], "r") as f:
            base_config = json.load(f)
        
        st.divider()

        with st.expander("💾 Dados (Fine-tuning)", expanded=True):
            st.markdown("**Valores detectados do modelo original**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                symbol_ft = st.selectbox(
                    "Symbol", 
                    options=AVAILABLE_SYMBOLS,
                    index=AVAILABLE_SYMBOLS.index(base_config.get("symbol", "BTCUSDT")) if base_config.get("symbol") in AVAILABLE_SYMBOLS else 0,
                    key="ft_symbol"
                )
            with col2:
                interval_ft = st.selectbox(
                    "Interval",
                    options=AVAILABLE_INTERVALS,
                    index=AVAILABLE_INTERVALS.index(base_config.get("interval", "1h")) if base_config.get("interval") in AVAILABLE_INTERVALS else 3,
                    key="ft_interval"
                )
            with col3:
                currency_ft = st.selectbox(
                    "Currency",
                    options=AVAILABLE_CURRENCIES,
                    index=AVAILABLE_CURRENCIES.index(base_config.get("currency", "USDT")) if base_config.get("currency") in AVAILABLE_CURRENCIES else 0,
                    key="ft_currency"
                )
            
            col4, col5 = st.columns(2)
            with col4:
                exchange_opts_ft = ["Todas"] + AVAILABLE_EXCHANGES
                default_exch = base_config.get("exchange", "Todas")
                if default_exch not in exchange_opts_ft:
                    default_exch = "Todas"
                exchange_ft = st.selectbox(
                    "Exchange",
                    options=exchange_opts_ft,
                    index=exchange_opts_ft.index(default_exch),
                    key="ft_exchange"
                )
            with col5:
                source_opts_ft = ["Todas"] + AVAILABLE_SOURCES
                default_src = base_config.get("source", "Todas")
                if default_src not in source_opts_ft:
                    default_src = "Todas"
                source_ft = st.selectbox(
                    "Source",
                    options=source_opts_ft,
                    index=source_opts_ft.index(default_src),
                    key="ft_source"
                )
            
            exchange_value_ft = None if exchange_ft == "Todas" else exchange_ft
            source_value_ft = None if source_ft == "Todas" else source_ft
            
            
            st.info(
                f"🔍 Fine-tuning usará dados de: **{symbol_ft}** ({interval_ft}) "
                f"em **{currency_ft}**"
            )
        
        # === CONFIGURAÇÃO DO FINE-TUNING ===
        with st.expander("⚙️ Configuração do Fine-tuning", expanded=True):
            st.markdown("**Ajuste apenas os parâmetros que deseja modificar**")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Data Inicial",
                    value=datetime.datetime.strptime(base_config.get("start_date", "2017-08-18 00:00:00"), "%Y-%m-%d %H:%M:%S"),
                    key="ft_start"
                )
                train_size = st.number_input("Train Size", value=base_config.get("train_size", 0.7), 
                                            min_value=0.01, max_value=0.99, step=0.01, format="%.2f")
                window_size = st.number_input("Window Size", value=base_config.get("window_size", 96), key="ft_window")
                epochs = st.number_input("Epochs (Fine-tune)", value=10, min_value=1, max_value=100, key="ft_epochs",
                                        help="Menos epochs para fine-tuning (evita overfitting)")
            
            with col2:
                end_date = st.date_input(
                    "Data Final",
                    value=datetime.datetime.strptime(base_config.get("end_date", "2025-01-19 23:59:59"), "%Y-%m-%d %H:%M:%S"),
                    key="ft_end"
                )
                validation_split = st.number_input("Validation Split", value=base_config.get("validation_split", 0.15),
                                                  min_value=0.01, max_value=0.99, step=0.01, format="%.2f")
                batch_size = st.number_input("Batch Size", value=base_config.get("batch_size", 32), key="ft_batch")
                patience = st.number_input("Patience", value=5, min_value=1, max_value=20, key="ft_patience")
            
            st.markdown("**Hiperparâmetros de Treino**")
            col3, col4 = st.columns(2)
            with col3:
                learning_rate = st.number_input("Learning Rate", value=0.0001, format="%.5f", key="ft_lr",
                                               help="Use LR menor para fine-tuning")
                dropout = st.number_input("Dropout", value=base_config.get("dropout", 0.2), format="%.2f", key="ft_dropout")
            with col4:
                optimizer = st.selectbox("Optimizer", ["Adam", "RMSprop", "SGD"], 
                                        index=_index_or_default(["Adam", "RMSprop", "SGD"], base_config.get("optimizer", "Adam")))
                loss_fn = st.selectbox("Loss Function", ["mean_squared_error", "mean_absolute_error", "mse"],
                                      index=_index_or_default(["mean_squared_error", "mean_absolute_error", "mse"], 
                                                             base_config.get("loss_fn", "mean_squared_error")))
        
        # === NORMALIZAÇÃO ===
        st.markdown("### Normalização")
        norm_defaults = base_config.get("normalization", {"strategy": "global", "scaler_type": "robust"})
        norm_cfg_ft = normalization_ui(defaults=norm_defaults, key_prefix="finetune")
        
        # === FEATURES E TARGET ===
        with st.expander("📋 Features e Target", expanded=False):
            target_column = st.selectbox(
                "Target Column",
                options=base_config.get("relevant_columns", ["close"]),
                index=_index_or_default(base_config.get("relevant_columns", ["close"]), 
                                       base_config.get("target_column", "close")),
                key="ft_target"
            )
            relevant_columns = st.multiselect(
                "Features (relevant_columns)", 
                options=base_config.get("relevant_columns", []),
                default=base_config.get("relevant_columns", []),
                key="ft_features"
            )
            steps_ahead = st.number_input("Steps Ahead", value=base_config.get("steps_ahead", 1),
                                         min_value=1, max_value=50, key="ft_steps")
        
        # === SEED ===
        use_seed = st.checkbox("Usar seed fixa", value="seed" in base_config, key="ft_use_seed")
        if use_seed:
            seed = st.number_input("Seed", min_value=0, max_value=2**32-1, 
                                  value=int(base_config.get("seed", 42)), step=1, key="ft_seed")
        else:
            seed = None
        
        # === PARÂMETROS ESPECÍFICOS DO MODELO ===
        with st.expander("🔧 Arquitetura do Modelo (opcional)", expanded=False):
            st.info("Deixe como está para manter a arquitetura original. Alterações aqui reconstruirão o modelo.")
            
            if selected_model["model_type"].lower() == "lstm":
                layers_config = st.text_input(
                    "Layers Config", 
                    value=",".join(str(x) for x in base_config.get("layers_config", [128, 64])),
                    key="ft_layers"
                )
                bidirectional = st.checkbox("Bidirectional", value=base_config.get("bidirectional", False), key="ft_bidir")
                l1_reg = st.number_input("L1 Reg", value=base_config.get("l1_reg", 0.0), key="ft_l1")
                l2_reg = st.number_input("L2 Reg", value=base_config.get("l2_reg", 0.0), key="ft_l2")
                activation_functions = st.text_input(
                    "Activation Functions", 
                    value=",".join(base_config.get("activation_functions", ["tanh", "tanh"])),
                    key="ft_act"
                )
                recurrent_dropout = st.number_input("Recurrent Dropout", value=base_config.get("recurrent_dropout", 0.0),
                                                   format="%.2f", key="ft_rec_drop")
            
            elif selected_model["model_type"].lower() == "transformer":
                num_layers = st.number_input("Num Layers", value=base_config.get("num_layers", 2), key="ft_num_layers")
                embed_dim = st.number_input("Embed Dim", value=base_config.get("embed_dim", 32), key="ft_embed")
                num_heads = st.number_input("Num Heads", value=base_config.get("num_heads", 2), key="ft_heads")
                ff_dim = st.number_input("FF Dim", value=base_config.get("ff_dim", 64), key="ft_ff")
                activation = st.text_input("Activation", value=base_config.get("activation", "relu"), key="ft_act_tr")
                l1_reg = st.number_input("L1 Reg", value=base_config.get("l1_reg", 0.0), key="ft_l1_tr")
                l2_reg = st.number_input("L2 Reg", value=base_config.get("l2_reg", 0.0), key="ft_l2_tr")
        
        st.divider()
        
        # === BOTÃO DE EXECUÇÃO ===
        col_btn1, col_btn2 = st.columns([3, 1])
        with col_btn1:
            execute_finetune = st.button("🚀 Executar Fine-tuning", type="primary", use_container_width=True, key="btn_exec_ft")
        with col_btn2:
            show_config = st.checkbox("Mostrar config", value=False, key="ft_show_cfg")
        
        if execute_finetune:
            # Montar config
            new_config = base_config.copy()
            new_config["window_size"] = int(window_size)
            new_config["batch_size"] = int(batch_size)
            new_config["epochs"] = int(epochs)
            new_config["patience"] = int(patience)
            new_config["learning_rate"] = float(learning_rate)
            new_config["dropout"] = float(dropout)
            new_config["optimizer"] = optimizer
            new_config["loss_fn"] = loss_fn
            new_config["steps_ahead"] = int(steps_ahead)
            new_config["target_column"] = target_column
            new_config["relevant_columns"] = list(relevant_columns)
            new_config["train_size"] = float(train_size)
            new_config["validation_split"] = float(validation_split)
            new_config["start_date"] = start_date.strftime("%Y-%m-%d 00:00:00")
            new_config["end_date"] = end_date.strftime("%Y-%m-%d 23:59:59")
            new_config["output_units"] = int(steps_ahead)
            new_config["run_id"] = str(uuid.uuid4())[:8]
            new_config["normalization"] = norm_cfg_ft
            new_config["symbol"] = symbol_ft
            new_config["interval"] = interval_ft
            new_config["currency"] = currency_ft
            
            if exchange_value_ft:
                new_config["exchange"] = exchange_value_ft
            if source_value_ft:
                new_config["source"] = source_value_ft
            
            if use_seed:
                new_config["seed"] = int(seed)
            else:
                new_config.pop("seed", None)
            
            if use_gpu_ft:
                new_config["use_gpu"] = True
                new_config["gpu_index"] = gpu_index_ft
            else:
                new_config["use_gpu"] = False
                new_config["gpu_index"] = None
            
            # Parâmetros específicos
            if selected_model["model_type"].lower() == "lstm":
                new_config["layers_config"] = [int(x) for x in layers_config.split(",") if x.strip()]
                new_config["bidirectional"] = bool(bidirectional)
                new_config["l1_reg"] = float(l1_reg)
                new_config["l2_reg"] = float(l2_reg)
                new_config["activation_functions"] = [x.strip() for x in activation_functions.split(",")]
                new_config["recurrent_dropout"] = float(recurrent_dropout)
            elif selected_model["model_type"].lower() == "transformer":
                new_config["num_layers"] = int(num_layers)
                new_config["embed_dim"] = int(embed_dim)
                new_config["num_heads"] = int(num_heads)
                new_config["ff_dim"] = int(ff_dim)
                new_config["activation"] = activation
                new_config["l1_reg"] = float(l1_reg)
                new_config["l2_reg"] = float(l2_reg)
            
            if show_config:
                st.json(new_config)

            if selected_model["framework"].lower() in ("keras", "tensorflow"):
                try:
                    import keras
                    keras.config.enable_unsafe_deserialization()
                except Exception:
                    pass

            
            # Executar
            with st.spinner("Executando fine-tuning... (aguarde, pode levar alguns minutos)"):
                try:
                    result = service.finetune(
                        model_path=selected_model["model_path"],
                        config=new_config,
                        framework=selected_model["framework"],
                        model_type=selected_model["model_type"],
                        original_run_id=selected_model["run_uuid"],
                        symbol=new_config["symbol"],
                        interval=new_config["interval"],
                        currency=new_config["currency"],
                        source=new_config.get("source"),
                        exchange=new_config.get("exchange"),
                    )
                    
                    st.success("Fine-tuning concluído com sucesso!")
                    
                    # Mostrar resultados
                    st.markdown("### Resultados")
                    
                    # Métricas
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Train Loss", f"{result.get('train_loss', 0):.4f}" if result.get('train_loss') else "N/A")
                    with col2:
                        st.metric("Val Loss", f"{result.get('val_loss', 0):.4f}" if result.get('val_loss') else "N/A")
                    with col3:
                        st.metric("Run ID", result.get('run_id', 'N/A'))
                    
                    # Comparação se disponível
                    if result.get('comparison'):
                        comp = result['comparison']
                        st.markdown("### Comparação com Modelo Original")
                        
                        if comp.get('metrics_comparison'):
                            metrics_comp = comp['metrics_comparison']
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                rmse_comp = metrics_comp.get('rmse', {})
                                delta_rmse = rmse_comp.get('delta', 0)
                                st.metric(
                                    "RMSE (Test)", 
                                    f"{rmse_comp.get('finetuned', 0):.4f}",
                                    delta=f"{delta_rmse:.4f}",
                                    delta_color="inverse"  # Menor é melhor
                                )
                                if rmse_comp.get('improvement_pct') is not None:
                                    st.caption(f"Melhoria: {rmse_comp['improvement_pct']:.2f}%")
                            
                            with col2:
                                mae_comp = metrics_comp.get('mae', {})
                                delta_mae = mae_comp.get('delta', 0)
                                st.metric(
                                    "MAE (Test)",
                                    f"{mae_comp.get('finetuned', 0):.4f}",
                                    delta=f"{delta_mae:.4f}",
                                    delta_color="inverse"
                                )
                                if mae_comp.get('improvement_pct') is not None:
                                    st.caption(f"Melhoria: {mae_comp['improvement_pct']:.2f}%")
                            
                            with col3:
                                r2_comp = metrics_comp.get('r2', {})
                                delta_r2 = r2_comp.get('delta', 0)
                                st.metric(
                                    "R² (Test)",
                                    f"{r2_comp.get('finetuned', 0):.4f}",
                                    delta=f"{delta_r2:.4f}",
                                    delta_color="normal"  # Maior é melhor
                                )
                                if r2_comp.get('improvement_pct') is not None:
                                    st.caption(f"Melhoria: {r2_comp['improvement_pct']:.2f}%")
                        
                        # Status da performance
                        perf_status = comp.get('performance_status', 'UNKNOWN')
                        if perf_status == "IMPROVED":
                            st.success("Performance melhorou ou se manteve em relação ao modelo original")
                        elif perf_status == "DEGRADED":
                            st.warning("Performance degradou. Considere usar o backup do modelo original.")
                            st.info(f"Backup disponível em: {result.get('backup_path', 'N/A')}")
                        else:
                            st.info("Comparação com baseline não disponível")
                    
                    # Arquivos gerados
                    with st.expander("Arquivos Gerados"):
                        st.code(f"""
Modelo Fine-tuned: {result['model_path']}
Backup Original: {result.get('backup_path', 'N/A')}
CSV Métricas: {result['csv_path']}
Comparação: {result.get('comparison_path', 'N/A')}
Log: {result.get('log_path', 'N/A')}
                        """, language="text")
                    
                    # Download do log
                    if result.get('log_path') and os.path.exists(result['log_path']):
                        with open(result['log_path'], "r") as f:
                            log_content = f.read()
                        st.download_button(
                            "Download Log Completo",
                            data=log_content,
                            file_name=f"finetune_{result['run_id']}.log",
                            mime="text/plain"
                        )
                
                except Exception as e:
                    st.error(f"Erro durante fine-tuning: {type(e).__name__}: {str(e)}")
                    with st.expander("Stack Trace"):
                        import traceback
                        st.code(traceback.format_exc())

# ---------------------------------
# 4. LIVE RUN
# ---------------------------------
def run_live_once():
    return service.live_run_once(
        model_path=selected_model["model_path"],
        config_path=selected_model["config_path"],
        framework=selected_model["framework"],
        model_type=selected_model["model_type"],
        symbol=symbol,
        interval=interval,
    )

with tabs[3]:
    st.header("Execução Online (Live Run)")
    models = list_all_models()
    selected_model = model_dropdown(models, key="liverun_model")
    if selected_model:
        st.code(f"Modelo selecionado: {selected_model['model_path']}")
        st.write(f"Framework: {selected_model['framework']}, Tipo: {selected_model['model_type']}")
        st.write(f"Treinado em: {selected_model.get('created_at', '')}")
        symbol = st.text_input("Símbolo (ex: BTCUSDT)", value="BTCUSDT")
        interval = st.selectbox("Intervalo", ["1h", "4h", "1d"], index=0)
       
        if selected_model["framework"].lower() in ("keras", "tensorflow"):
            try:
                import keras
                keras.config.enable_unsafe_deserialization()
            except Exception:
                pass

        slot = st.empty()
        live_slot = st.empty()
        col1, col2 = st.columns(2)
        if col1.button("Iniciar Live Run", key="btn_start_live"):
            st.session_state["live_run_active"] = True
            st.session_state["last_live_run_result"] = None
            st.rerun()
        if col2.button("Parar Live Run", key="btn_stop_live"):
            st.session_state["live_run_active"] = False
            live_slot.write("Execução parada pelo usuário.")

        live_active = st.session_state.get("live_run_active", False)

        if live_active:
            try:
                result = run_live_once()
                st.session_state["last_live_run_result"] = result
                with live_slot:
                    result_txt = f"""
                    ### ⏰ Timestamp: `{result['timestamp']}`
                    **Preço real ({result['target_column']}):** `${result['close_real']:.2f}`  
                    """
                    if result["steps_ahead"] > 1:
                        preds = "\n".join([f"- T+{i+1}: `{pred:.2f}`" for i, pred in enumerate(result["previsoes"])])
                        result_txt += f"**Previsões:**\n{preds}"
                    else:
                        result_txt += f"**Previsão:** `{result['previsoes'][0]:.2f}`"
                    st.markdown(result_txt)
            except Exception as e:
                with live_slot:
                    st.error(f"Erro ao executar live run: {e}")

            time.sleep(60)
            st.rerun()

        elif st.session_state.get("last_live_run_result") is not None:
            result = st.session_state["last_live_run_result"]
            with live_slot:
                result_txt = f"""
                ## Execução parada. Último resultado:
                ### ⏰ Timestamp: `{result['timestamp']}`
                **Preço real ({result['target_column']}):** `${result['close_real']:.2f}`  
                """
                if result["steps_ahead"] > 1:
                    preds = "\n".join([f"- T+{i+1}: `{pred:.2f}`" for i, pred in enumerate(result["previsoes"])])
                    result_txt += f"**Previsões:**\n{preds}"
                else:
                    result_txt += f"**Previsão:** `{result['previsoes'][0]:.2f}`"
                st.markdown(result_txt)
