# app.py
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

# -- Parâmetros para opções
frameworks = ["keras", "pytorch", "tensorflow"]
model_types = ["lstm", "transformer"]
financial_columns = ["close", "open", "high", "low", "volume"]
indicator_options = [
    {"label": "SMA", "key": "sma", "param_label": "Período", "default": 14},
    {"label": "EMA", "key": "ema", "param_label": "Período", "default": 14},
]

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
    "Grid Search",
    "Treino Normal",
    "Fine-tuning",
    "Live Run"
])

# ---------------------------------
# 1. GRID SEARCH
# ---------------------------------
with tabs[0]:
    st.header("Grid Search de Hiperparâmetros")
    st.info("Monte seu grid de hiperparâmetros para múltiplos experimentos.")
    frameworks = ["keras", "pytorch", "tensorflow"]
    model_types = ["lstm", "transformer"]
    with st.form("grid_search_form"):
        st.subheader("Geral")
        framework = st.selectbox("Framework", frameworks)
        model_type = st.selectbox("Modelo", model_types)
        st.markdown("#### Hiperparâmetros para o Grid")
        window_sizes = st.text_input("Window Sizes (ex: 72,96,120)", value="72,96,120")
        batch_sizes = st.text_input("Batch Sizes (ex: 16,32,64)", value="16,32,64")
        epochs = st.text_input("Epochs (ex: 100,200)", value="100")
        patience = st.text_input("Patience (ex: 5,10)", value="5,10")
        learning_rates = st.text_input("Learning Rates (ex: 0.001,0.0005)", value="0.001,0.0005")
        dropout_rates = st.text_input("Dropout Rates (ex: 0.2,0.3,0.5)", value="0.2,0.3,0.5")
        optimizer = st.multiselect("Optimizer", options=["Adam", "RMSprop", "SGD"], default=["Adam"])
        loss_functions = st.multiselect("Função de Perda", options=["mean_squared_error", "mean_absolute_error", "mse"], default=["mean_squared_error"])

        if model_type == "lstm":
            layers_config = st.text_input("Layers Config (ex: [128,64],[64,64])", value="[128,64],[64,64]")
            bidirectional = st.multiselect("Bidirecional", options=[True, False], default=[False])
            l1 = st.text_input("L1 Regularization", value="0.0,0.001")
            l2 = st.text_input("L2 Regularization", value="0.0,0.001")
            activation = st.text_input("Função de Ativação (ex: tanh,relu)", value="tanh")
        else:  # transformer
            num_layers = st.text_input("Num Layers", value="2,3")
            embed_dim = st.text_input("Embed Dim", value="32,64")
            num_heads = st.text_input("Num Heads", value="2,4")
            ff_dim = st.text_input("FF Dim", value="64,128")
            activation = st.text_input("Função de Ativação (ex: relu,g elu)", value="relu")
            l1 = st.text_input("L1 Regularization", value="0.0,0.001")
            l2 = st.text_input("L2 Regularization", value="0.0,0.001")

        submit_btn = st.form_submit_button("Salvar Configuração de Grid")
        if submit_btn:
            hyperparams = {
                "FRAMEWORK": [framework],
                "MODEL_TYPE": [model_type],
                "WINDOW_SIZE": [int(x) for x in window_sizes.split(",") if x],
                "BATCH_SIZE": [int(x) for x in batch_sizes.split(",") if x],
                "EPOCHS": [int(x) for x in epochs.split(",") if x],
                "PATIENCE": [int(x) for x in patience.split(",") if x],
                "LEARNING_RATE": [float(x) for x in learning_rates.split(",") if x],
                "DROPOUT": [float(x) for x in dropout_rates.split(",") if x],
                "OPTIMIZER": optimizer,
                "LOSS_FUNCTION": loss_functions,
            }
            if model_type == "lstm":
                hyperparams["LAYERS_CONFIG"] = [json.loads(x + "]") if not x.strip().endswith("]") else json.loads(x) for x in layers_config.replace("],[", "]|[").split("|")]
                hyperparams["BIDIRECTIONAL"] = bidirectional
                hyperparams["L1_REGULARIZATION"] = [float(x) for x in l1.split(",") if x]
                hyperparams["L2_REGULARIZATION"] = [float(x) for x in l2.split(",") if x]
                hyperparams["ACTIVATION_FUNCTION"] = [x.strip() for x in activation.split(",")]
            else:
                hyperparams["NUM_LAYERS"] = [int(x) for x in num_layers.split(",") if x]
                hyperparams["EMBED_DIM"] = [int(x) for x in embed_dim.split(",") if x]
                hyperparams["NUM_HEADS"] = [int(x) for x in num_heads.split(",") if x]
                hyperparams["FF_DIM"] = [int(x) for x in ff_dim.split(",") if x]
                hyperparams["L1_REGULARIZATION"] = [float(x) for x in l1.split(",") if x]
                hyperparams["L2_REGULARIZATION"] = [float(x) for x in l2.split(",") if x]
                hyperparams["ACTIVATION"] = [x.strip() for x in activation.split(",")]
            unique_id = str(uuid.uuid4())[:8]
            st.success(f"Configuração de grid salva! ID: {unique_id}")
            st.download_button("Download Grid Config JSON", data=json.dumps(hyperparams, indent=4), file_name=f"grid_config_{unique_id}.json")

    st.markdown("### Configurações de Grid Salvas:")

# ---------------------------------
# 2. TREINO NORMAL
# ---------------------------------
with tabs[1]:
    st.header("Treinamento de Modelo Único")
    st.info("Defina e treine um modelo único com os hiperparâmetros desejados.")

    st.subheader("Escolha do Modelo e o Framework")
    framework = st.selectbox("Framework", frameworks, key="framework_train")
    model_type = st.selectbox("Modelo", model_types, key="model_type_train")
    use_gpu, gpu_index, gpu_label = gpu_selector(framework, key_prefix="train")

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
            }
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
# 3. FINE-TUNING
# ---------------------------------
with tabs[2]:
    st.header("Fine-tuning de Modelo")
    models = list_all_models()
    selected_model = model_dropdown(models, key="finetune_model")
    if selected_model is not None:
        use_gpu_ft, gpu_index_ft, gpu_label_ft = gpu_selector(selected_model["framework"], key_prefix="finetune")
        st.code(f"Modelo selecionado: {selected_model['model_path']}")
        with open(selected_model["config_path"], "r") as f:
            base_config = json.load(f)

        st.markdown("#### Hiperparâmetros do modelo (ajuste apenas o que quiser):")

        start_date = st.date_input(
            "Data Inicial dos dados",
            value=datetime.datetime.strptime(base_config.get("start_date", "2017-08-18 00:00:00"), "%Y-%m-%d %H:%M:%S")
        )
        end_date = st.date_input(
            "Data Final dos dados",
            value=datetime.datetime.strptime(base_config.get("end_date", "2025-01-19 23:59:59"), "%Y-%m-%d %H:%M:%S")
        )
        window_size = st.number_input("Window Size", value=base_config.get("window_size", 96))
        batch_size = st.number_input("Batch Size", value=base_config.get("batch_size", 32))
        epochs = st.number_input("Epochs (Fine-tune)", value=base_config.get("epochs", 10))
        patience = st.number_input("Patience", value=base_config.get("patience", 5))
        learning_rate = st.number_input("Learning Rate", value=base_config.get("learning_rate", 0.0001), format="%.5f")
        dropout = st.number_input("Dropout", value=base_config.get("dropout", 0.2), format="%.2f")
        optimizer = st.selectbox("Optimizer", ["Adam", "RMSprop", "SGD"], index=_index_or_default(["Adam", "RMSprop", "SGD"], base_config.get("optimizer", "Adam")))
        loss_fn = st.selectbox("Função de Perda", ["mean_squared_error", "mean_absolute_error", "mse"], index=_index_or_default(["mean_squared_error", "mean_absolute_error", "mse"], base_config.get("loss_fn", "mean_squared_error")))
        steps_ahead = st.number_input("Steps Ahead (outputs)", value=base_config.get("steps_ahead", 1), min_value=1, max_value=50)
        target_column = st.selectbox(
            "Coluna alvo (target column)",
            options=base_config.get("relevant_columns", ["close"]),
            index=_index_or_default(base_config.get("relevant_columns", ["close"]), base_config.get("target_column", "close")),
            key="finetune_target_column"
        )
        relevant_columns = st.multiselect("Colunas usadas como features", options=base_config.get("relevant_columns", []), default=base_config.get("relevant_columns", []))
        train_size = st.number_input("Train Size", value=base_config.get("train_size", 0.7), min_value=0.01, max_value=0.99, step=0.01, format="%.2f")
        validation_split = st.number_input("Validation Split", value=base_config.get("validation_split", 0.15), min_value=0.01, max_value=0.99, step=0.01, format="%.2f")

        use_seed = st.checkbox("Usar seed fixa para reprodução", value="seed" in base_config)
        if use_seed:
            seed = st.number_input("Seed (reprodutibilidade)", min_value=0, max_value=2**32-1, value=int(base_config.get("seed", 42)), step=1)
        else:
            seed = None

        st.markdown("### Normalização")
        norm_defaults = base_config.get("normalization", {"strategy": "global", "scaler_type": "robust"})
        norm_cfg_ft = normalization_ui(defaults=norm_defaults, key_prefix="finetune")

        # Campos ESPECÍFICOS por tipo de modelo
        if selected_model["model_type"].lower() == "lstm":
            layers_config = st.text_input("Layers Config (ex: 128,64)", value=",".join(str(x) for x in base_config.get("layers_config", [128, 64])))
            bidirectional = st.checkbox("Bidirecional", value=base_config.get("bidirectional", False))
            l1_reg = st.number_input("L1 Regularization", value=base_config.get("l1_reg", 0.0))
            l2_reg = st.number_input("L2 Regularization", value=base_config.get("l2_reg", 0.0))
            activation_functions = st.text_input("Funções de Ativação (ex: tanh,relu)", value=",".join(base_config.get("activation_functions", ["tanh", "tanh"])))
            recurrent_dropout = st.number_input("Recurrent Dropout", value=base_config.get("recurrent_dropout", 0.0), format="%.2f")
        elif selected_model["model_type"].lower() == "transformer":
            num_layers = st.number_input("Num Layers", value=base_config.get("num_layers", 2))
            embed_dim = st.number_input("Embed Dim", value=base_config.get("embed_dim", 32))
            num_heads = st.number_input("Num Heads", value=base_config.get("num_heads", 2))
            ff_dim = st.number_input("FF Dim", value=base_config.get("ff_dim", 64))
            activation = st.text_input("Função de Ativação", value=base_config.get("activation", "relu"))
            l1_reg = st.number_input("L1 Regularization", value=base_config.get("l1_reg", 0.0))
            l2_reg = st.number_input("L2 Regularization", value=base_config.get("l2_reg", 0.0))
        else:
            st.warning("Tipo de modelo não suportado neste bloco!")

        if st.button("Executar Fine-tuning", key="btn_finetune"):
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

            with st.spinner("Executando fine-tuning..."):
                result = service.finetune(
                    model_path=selected_model["model_path"],
                    config=new_config,
                    framework=selected_model["framework"],
                    model_type=selected_model["model_type"],
                    original_run_id=selected_model["run_uuid"],
                )
            st.success(f"Fine-tuning concluído! Caminho: {result['model_path']}")
            st.code(json.dumps(result, indent=2))

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
