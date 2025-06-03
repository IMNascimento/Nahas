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
    # Adicione mais se quiser...
]


# -- Session state para sincronização
if "selected_financial" not in st.session_state:
    st.session_state.selected_financial = list(financial_columns)
if "indicators_apply" not in st.session_state:
    st.session_state.indicators_apply = {}
if "indicator_columns" not in st.session_state:
    st.session_state.indicator_columns = []
if "relevant_columns" not in st.session_state:
    st.session_state.relevant_columns = list(financial_columns)
if "live_run_active" not in st.session_state:
    st.session_state.live_run_active = False
if "last_live_run_result" not in st.session_state:
    st.session_state.last_live_run_result = None


def update_feature_list():
    # Atualiza colunas possíveis (financeiras + indicadores ativos)
    st.session_state.indicator_columns = []
    st.session_state.indicators_apply = {}
    for ind in indicator_options:
        checked = st.session_state.get(f"checked_{ind['key']}", False)
        if checked:
            period = st.session_state.get(f"param_{ind['key']}", ind["default"])
            col_name = f"{ind['key']}_{period}"
            st.session_state.indicator_columns.append(col_name)
            st.session_state.indicators_apply[ind["key"]] = [{"period": period, "col_name": col_name}]
    # Atualiza lista de todas as possíveis features
    all_cols = st.session_state.selected_financial + st.session_state.indicator_columns
    st.session_state.all_possible_columns = all_cols
    # Remove das features qualquer coluna que não existe mais
    st.session_state.relevant_columns = [
        col for col in st.session_state.relevant_columns if col in all_cols
    ]


st.set_page_config(
    page_title="Nahas - Plataforma de Modelos Financeiros",
    layout="wide",
    page_icon=":chart_with_upwards_trend:",
)

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
# 1. GRID SEARCH (todos hiperparâmetros)
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
        # Parâmetros comuns
        window_sizes = st.text_input("Window Sizes (ex: 72,96,120)", value="72,96,120")
        batch_sizes = st.text_input("Batch Sizes (ex: 16,32,64)", value="16,32,64")
        epochs = st.text_input("Epochs (ex: 100,200)", value="100")
        patience = st.text_input("Patience (ex: 5,10)", value="5,10")
        learning_rates = st.text_input("Learning Rates (ex: 0.001,0.0005)", value="0.001,0.0005")
        dropout_rates = st.text_input("Dropout Rates (ex: 0.2,0.3,0.5)", value="0.2,0.3,0.5")
        optimizer = st.multiselect("Optimizer", options=["Adam", "RMSprop", "SGD"], default=["Adam"])
        loss_functions = st.multiselect("Função de Perda", options=["mean_squared_error", "mean_absolute_error", "mse"], default=["mean_squared_error"])
        # Model-specific (exemplo para lstm/transformer)
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
            # Monta hiperparâmetros em grid para salvar
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
    st.subheader("Colunas Financeiras e alvos(target)")
    indicator_columns = []
    indicators_apply = {}

    with st.expander("Indicadores Técnicos"):
        if st.checkbox("SMA"):
            sma_period = st.number_input("SMA - Período", value=14, key="sma_period")
            sma_col_name = f"sma_{sma_period}"
            indicators_apply["sma"] = [{"period": sma_period, "col_name": sma_col_name}]
            indicator_columns.append(sma_col_name)
        if st.checkbox("EMA"):
            ema_period = st.number_input("EMA - Período", value=14, key="ema_period")
            ema_col_name = f"ema_{ema_period}"
            indicators_apply["ema"] = [{"period": ema_period, "col_name": ema_col_name}]
            indicator_columns.append(ema_col_name)
         # Envelopes
        if st.checkbox("Envelopes"):
            env_period = st.number_input("Envelopes - Período", value=14, key="envelopes_period")
            env_percent = st.number_input("Envelopes - Percentual", value=3.0, key="envelopes_percent")
            env_col_name = f"envelopes_{env_period}_{env_percent}"
            indicators_apply["envelopes"] = [{"period": env_period, "percent": env_percent, "col_name": env_col_name}]
            indicator_columns.append(env_col_name)
        # RSI
        if st.checkbox("RSI"):
            rsi_period = st.number_input("RSI - Período", value=14, key="rsi_period")
            rsi_col_name = f"rsi_{rsi_period}"
            indicators_apply["rsi"] = [{"period": rsi_period, "col_name": rsi_col_name}]
            indicator_columns.append(rsi_col_name)
        # MACD
        if st.checkbox("MACD"):
            fastperiod = st.number_input("MACD - FastPeriod", value=12, key="macd_fast")
            slowperiod = st.number_input("MACD - SlowPeriod", value=26, key="macd_slow")
            signalperiod = st.number_input("MACD - SignalPeriod", value=9, key="macd_signal")
            macd_col_name = f"macd_{fastperiod}_{slowperiod}_{signalperiod}"
            indicators_apply["macd"] = [{
                "fastperiod": fastperiod, "slowperiod": slowperiod,
                "signalperiod": signalperiod, "col_name": macd_col_name
            }]
            indicator_columns.append(macd_col_name)
        # Bollinger Bands
        if st.checkbox("Bollinger Bands"):
            bb_period = st.number_input("Bollinger Bands - Período", value=20, key="bb_period")
            bb_stddev = st.number_input("Bollinger Bands - Std Dev", value=2.0, key="bb_stddev")
            bb_col_name = f"bollinger_bands_{bb_period}_{bb_stddev}"
            indicators_apply["bollinger_bands"] = [{"period": bb_period, "std_dev": bb_stddev, "col_name": bb_col_name}]
            indicator_columns.append(bb_col_name)
        # ATR
        if st.checkbox("ATR"):
            atr_period = st.number_input("ATR - Período", value=14, key="atr_period")
            atr_col_name = f"atr_{atr_period}"
            indicators_apply["atr"] = [{"period": atr_period, "col_name": atr_col_name}]
            indicator_columns.append(atr_col_name)
        # ADX
        if st.checkbox("ADX"):
            adx_period = st.number_input("ADX - Período", value=14, key="adx_period")
            adx_col_name = f"adx_{adx_period}"
            indicators_apply["adx"] = [{"period": adx_period, "col_name": adx_col_name}]
            indicator_columns.append(adx_col_name)
        # Stochastic
        if st.checkbox("Stochastic"):
            stoch_period = st.number_input("Stochastic - Período", value=14, key="stoch_period")
            stoch_col_name = f"stochastic_{stoch_period}"
            indicators_apply["stochastic"] = [{"period": stoch_period, "col_name": stoch_col_name}]
            indicator_columns.append(stoch_col_name)
        # Fibonacci Retracement
        if st.checkbox("Fibonacci Retracement"):
            fib_ret_col_name = "fibonacci_retracement"
            indicators_apply["fibonacci_retracement"] = [{}]
            indicator_columns.append(fib_ret_col_name)
        # Fibonacci Projection
        if st.checkbox("Fibonacci Projection"):
            fib_proj_col_name = "fibonacci_projection"
            indicators_apply["fibonacci_projection"] = [{}]
            indicator_columns.append(fib_proj_col_name)
            # ...mais indicadores...

    financial_columns = ["close", "open", "high", "low", "volume"]
    all_possible_columns = financial_columns + indicator_columns
    # 2. Escolha da coluna alvo (target)
    target_column = st.selectbox(
        "Coluna alvo (target column)",
        options=all_possible_columns,
        index=all_possible_columns.index("close") if "close" in all_possible_columns else 0
    )
    # 3. Só um multiselect para features finais
    relevant_columns = st.multiselect(
        "Colunas finais usadas no modelo (features)",
        options=all_possible_columns,
        default=all_possible_columns
    )
    use_seed = st.checkbox("Usar seed fixa para reprodução", value=False)

    with st.form("train_form"):
        st.subheader("Hiperparâmetros")
        
        if use_seed:
            seed = st.number_input("Seed (reprodutibilidade)", min_value=0, max_value=9999999999, value=42, step=1)
        else:
            seed = None  # Será aleatória (ou não incluída)
        start_date = st.date_input("Data Inicial dos dados", value=datetime.date(2017, 8 , 18 ))
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
        steps_ahead = st.number_input(
            "Steps Ahead (nº de passos à frente para previsão) e é Output Units (nº de saídas do modelo)",
            value=1, min_value=1, max_value=50
        )
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
                "window_size": window_size,
                "batch_size": batch_size,
                "epochs": epochs,
                "patience": patience,
                "learning_rate": learning_rate,
                "dropout": dropout,
                "optimizer": optimizer,
                "loss_fn": loss_function,
                "start_date": start_date.strftime("%Y-%m-%d 00:00:00"),
                "end_date": end_date.strftime("%Y-%m-%d 23:59:59"),
                "relevant_columns": relevant_columns,
                "target_column": target_column,
                "indicators_apply": indicators_apply,
                "train_size": train_size,
                "validation_split": validation_split,
                "steps_ahead": steps_ahead,
                "output_units": int(steps_ahead), 
                "run_id": str(uuid.uuid4())[:8],
            }
            if model_type == "lstm":
                config["layers_config"] = [int(x) for x in layers.split(",")]
                config["bidirectional"] = bidirectional
                config["l1_reg"] = l1
                config["l2_reg"] = l2
                config["activation_functions"] = [x.strip() for x in activation.split(",")]
                config["recurrent_dropout"] = recurrent_dropout
            else:
                config["num_layers"] = num_layers
                config["embed_dim"] = embed_dim
                config["num_heads"] = num_heads
                config["ff_dim"] = ff_dim
                config["activation"] = activation
                config["l1_reg"] = l1
                config["l2_reg"] = l2
            
            if use_gpu:
                config["use_gpu"] = True
                config["gpu_index"] = gpu_index
            else:
                config["use_gpu"] = False
                config["gpu_index"] = None
            if use_seed:
                config["seed"] = int(seed)
            # Chama service universal
            st.info("Treinando modelo, aguarde...")
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
    use_gpu_ft, gpu_index_ft, gpu_label_ft = gpu_selector(selected_model["framework"], key_prefix="finetune")

    if selected_model:
        st.code(f"Modelo selecionado: {selected_model['model_path']}")
        with open(selected_model["config_path"], "r") as f:
            config = json.load(f)
        
        st.markdown("#### Hiperparâmetros do modelo (ajuste apenas o que quiser):")

        # Campos comuns
        start_date = st.date_input("Data Inicial dos dados", value=datetime.datetime.strptime(config.get("start_date", "2017-08-18 00:00:00"), "%Y-%m-%d %H:%M:%S"))
        end_date = st.date_input("Data Final dos dados", value=datetime.datetime.strptime(config.get("end_date", "2025-01-19 23:59:59"), "%Y-%m-%d %H:%M:%S"))
        window_size = st.number_input("Window Size", value=config.get("window_size", 96))
        batch_size = st.number_input("Batch Size", value=config.get("batch_size", 32))
        epochs = st.number_input("Epochs (Fine-tune)", value=config.get("epochs", 10))
        patience = st.number_input("Patience", value=config.get("patience", 5))
        learning_rate = st.number_input("Learning Rate", value=config.get("learning_rate", 0.0001), format="%.5f")
        dropout = st.number_input("Dropout", value=config.get("dropout", 0.2), format="%.2f")
        optimizer = st.selectbox("Optimizer", ["Adam", "RMSprop", "SGD"], index=["Adam", "RMSprop", "SGD"].index(config.get("optimizer", "Adam")))
        loss_fn = st.selectbox("Função de Perda", ["mean_squared_error", "mean_absolute_error", "mse"], index=["mean_squared_error", "mean_absolute_error", "mse"].index(config.get("loss_fn", "mean_squared_error")))
        steps_ahead = st.number_input("Steps Ahead (outputs)", value=config.get("steps_ahead", 1), min_value=1, max_value=50)
        target_column = st.selectbox(
            "Coluna alvo (target column)",
            options=config.get("relevant_columns", ["close"]),
            index=config.get("relevant_columns", ["close"]).index(config.get("target_column", "close")),
            key="finetune_target_column"   # chave única aqui!
        )
        relevant_columns = st.multiselect("Colunas usadas como features", options=config.get("relevant_columns", []), default=config.get("relevant_columns", []))
        train_size = st.number_input("Train Size", value=config.get("train_size", 0.7), min_value=0.01, max_value=0.99, step=0.01, format="%.2f")
        validation_split = st.number_input("Validation Split", value=config.get("validation_split", 0.15), min_value=0.01, max_value=0.99, step=0.01, format="%.2f")
        # Seed (opcional)
        use_seed = st.checkbox("Usar seed fixa para reprodução", value="seed" in config)
        if use_seed:
            seed = st.number_input("Seed (reprodutibilidade)", min_value=0, max_value=9999999999, value=int(config.get("seed", 42)), step=1)
        else:
            seed = None

        # Campos ESPECÍFICOS por tipo de modelo
        if selected_model["model_type"].lower() == "lstm":
            layers_config = st.text_input("Layers Config (ex: 128,64)", value=",".join(str(x) for x in config.get("layers_config", [128, 64])))
            bidirectional = st.checkbox("Bidirecional", value=config.get("bidirectional", False))
            l1_reg = st.number_input("L1 Regularization", value=config.get("l1_reg", 0.0))
            l2_reg = st.number_input("L2 Regularization", value=config.get("l2_reg", 0.0))
            activation_functions = st.text_input("Funções de Ativação (ex: tanh,relu)", value=",".join(config.get("activation_functions", ["tanh", "tanh"])))
            recurrent_dropout = st.number_input("Recurrent Dropout", value=config.get("recurrent_dropout", 0.0), format="%.2f")
        elif selected_model["model_type"].lower() == "transformer":
            num_layers = st.number_input("Num Layers", value=config.get("num_layers", 2))
            embed_dim = st.number_input("Embed Dim", value=config.get("embed_dim", 32))
            num_heads = st.number_input("Num Heads", value=config.get("num_heads", 2))
            ff_dim = st.number_input("FF Dim", value=config.get("ff_dim", 64))
            activation = st.text_input("Função de Ativação", value=config.get("activation", "relu"))
            l1_reg = st.number_input("L1 Regularization", value=config.get("l1_reg", 0.0))
            l2_reg = st.number_input("L2 Regularization", value=config.get("l2_reg", 0.0))
        else:
            st.warning("Tipo de modelo não suportado neste bloco!")

        # --- Executa Fine-tuning ---
        if st.button("Executar Fine-tuning"):
            # Novo config (permite sobrescrever só os campos do form)
            new_config = config.copy()
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
            new_config["relevant_columns"] = relevant_columns
            new_config["train_size"] = float(train_size)
            new_config["validation_split"] = float(validation_split)
            new_config["start_date"] = start_date.strftime("%Y-%m-%d 00:00:00")
            new_config["end_date"] = end_date.strftime("%Y-%m-%d 23:59:59")
            new_config["output_units"] = int(steps_ahead)
            new_config["run_id"] = str(uuid.uuid4())[:8]
            if use_seed:
                new_config["seed"] = int(seed)
            else:
                if "seed" in new_config:
                    del new_config["seed"]
            if use_gpu_ft:
                new_config["use_gpu"] = True
                new_config["gpu_index"] = gpu_index_ft
            else:
                new_config["use_gpu"] = False
                new_config["gpu_index"] = None

            # Campos específicos LSTM
            if selected_model["model_type"].lower() == "lstm":
                new_config["layers_config"] = [int(x) for x in layers_config.split(",") if x]
                new_config["bidirectional"] = bidirectional
                new_config["l1_reg"] = float(l1_reg)
                new_config["l2_reg"] = float(l2_reg)
                new_config["activation_functions"] = [x.strip() for x in activation_functions.split(",")]
                new_config["recurrent_dropout"] = float(recurrent_dropout)
            # Campos específicos Transformer
            elif selected_model["model_type"].lower() == "transformer":
                new_config["num_layers"] = int(num_layers)
                new_config["embed_dim"] = int(embed_dim)
                new_config["num_heads"] = int(num_heads)
                new_config["ff_dim"] = int(ff_dim)
                new_config["activation"] = activation
                new_config["l1_reg"] = float(l1_reg)
                new_config["l2_reg"] = float(l2_reg)

            # Treinamento!
            result = service.finetune(
                model_path=selected_model["model_path"],         # Caminho do modelo base
                config=new_config,                               # Novo config
                framework=selected_model["framework"],           # Framework (keras, etc)
                model_type=selected_model["model_type"],         # Modelo (lstm, transformer)
                original_run_id=selected_model["run_uuid"],        # Relaciona com o treino original!
            )

            st.success(f"Fine-tuning concluído! Caminho: {result['model_path']}")
            st.code(json.dumps(result, indent=2))


# ---------------------------------
# 4. LIVE RUN
# ---------------------------------

def run_live_once():
    # Chame aqui o seu método de predição real!
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
        if col1.button("Iniciar Live Run"):
            st.session_state.live_run_active = True
            st.session_state.last_live_run_result = None  # Limpa resultado anterior
            st.rerun() # Garante início imediato
        if col2.button("Parar Live Run"):
            st.session_state.live_run_active = False
            live_slot.write("Execução parada pelo usuário.")

        # Loop baseado em session_state
        if st.session_state.live_run_active:
            try:
                result = run_live_once()
                st.session_state.last_live_run_result = result  # Salva último resultado para exibir depois
                
                 # Exibe resultado sempre no mesmo slot!
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

            # Aguarda X segundos e força novo ciclo
            time.sleep(60)
            st.rerun()

        elif st.session_state.last_live_run_result is not None:
            # Exibe último resultado enquanto parado
            result = st.session_state.last_live_run_result
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