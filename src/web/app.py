import sys
import os
import streamlit as st
import json
import uuid
from db_helpers import list_all_models
from components.model_dropdown import model_dropdown

from config.settings import Settings
from services.model_service import ModelService

service = ModelService()
GENERATED_CONFIGS_PATH = "../src/config/generated"
os.makedirs(GENERATED_CONFIGS_PATH, exist_ok=True)

def save_hyperparams(hyperparams, prefix="hyperparameters"):
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{prefix}_{unique_id}.json"
    path = os.path.join(GENERATED_CONFIGS_PATH, filename)
    with open(path, "w") as f:
        json.dump(hyperparams, f, indent=4)
    return path, unique_id

def list_configs(folder=GENERATED_CONFIGS_PATH):
    return [f for f in os.listdir(folder) if f.endswith('.json')]

def load_config(path):
    with open(path, "r") as f:
        return json.load(f)

st.set_page_config(
    page_title="Nahas - Plataforma de Modelos Financeiros",
    layout="wide",
    page_icon=":chart_with_upwards_trend:",
)

with st.sidebar:
    st.title("Nahas ML Platform")
    st.markdown("Powered by Streamlit")
    st.markdown("### Acesso Rápido")
    st.markdown("- [Documentação](../docs/README.md)")
    st.markdown("- [Contato/Suporte](mailto:suporte@nahas.com)")

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
            path, unique_id = save_hyperparams(hyperparams, prefix="gridsearch")
            st.success(f"Configuração de grid salva! ID: {unique_id}")
            st.download_button("Download Grid Config JSON", data=json.dumps(hyperparams, indent=4), file_name=f"grid_config_{unique_id}.json")

    st.markdown("### Configurações de Grid Salvas:")
    for cfg in list_configs():
        if cfg.startswith("gridsearch_"):
            st.write(cfg)

# ---------------------------------
# 2. TREINO NORMAL
# ---------------------------------
with tabs[1]:
    st.header("Treinamento de Modelo Único")
    st.info("Defina e treine um modelo único com os hiperparâmetros desejados.")
    frameworks = ["keras", "pytorch", "tensorflow"]
    model_types = ["lstm", "transformer"]
    with st.form("train_form"):
        st.subheader("Hiperparâmetros")
        framework = st.selectbox("Framework", frameworks, key="framework_train")
        model_type = st.selectbox("Modelo", model_types, key="model_type_train")
        window_size = st.number_input("Window Size", value=96)
        batch_size = st.number_input("Batch Size", value=32)
        epochs = st.number_input("Epochs", value=100)
        patience = st.number_input("Patience", value=5)
        learning_rate = st.number_input("Learning Rate", value=0.001, format="%.5f")
        dropout = st.number_input("Dropout", value=0.2, format="%.2f")
        optimizer = st.selectbox("Optimizer", ["Adam", "RMSprop", "SGD"])
        loss_function = st.selectbox("Função de Perda", ["mean_squared_error", "mean_absolute_error", "mse"])
        if model_type == "lstm":
            layers = st.text_input("Layers (ex: 128,64)", value="128,64")
            bidirectional = st.checkbox("Bidirecional", value=False)
            l1 = st.number_input("L1 Regularization", value=0.0)
            l2 = st.number_input("L2 Regularization", value=0.0)
            activation = st.text_input("Função de Ativação (ex: tanh,relu)", value="tanh")
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
                "loss_fn": loss_function
            }
            if model_type == "lstm":
                config["layers_config"] = [int(x) for x in layers.split(",")]
                config["bidirectional"] = bidirectional
                config["l1_reg"] = l1
                config["l2_reg"] = l2
                config["activation_functions"] = [x.strip() for x in activation.split(",")]
            else:
                config["num_layers"] = num_layers
                config["embed_dim"] = embed_dim
                config["num_heads"] = num_heads
                config["ff_dim"] = ff_dim
                config["activation"] = activation
                config["l1_reg"] = l1
                config["l2_reg"] = l2
            # Chama service universal
            st.info("Treinando modelo, aguarde...")
            result = service.train(config, framework, model_type)
            st.success(f"Modelo treinado! Caminho: {result['model_path']}")
            st.code(json.dumps(result, indent=2))

# ---------------------------------
# 3. FINE-TUNING
# ---------------------------------
with tabs[2]:
    st.header("Fine-tuning")
    models = list_all_models()
    selected_model = model_dropdown(models, key="finetune_model")
    if selected_model:
        st.code(f"Modelo selecionado: {selected_model['model_path']}")
        epochs = st.number_input("Epochs (Fine-tune)", value=10)
        learning_rate = st.number_input("Learning Rate (Fine-tune)", value=0.001)
        ft_btn = st.button("Executar Fine-tuning")
        if ft_btn:
            st.info("Executando fine-tuning (em breve: integração total)")
            # service.finetune(...)  # TODO: Adicionar chamada

# ---------------------------------
# 4. LIVE RUN
# ---------------------------------
with tabs[3]:
    st.header("Execução Online (Live Run)")
    models = list_all_models()
    selected_model = model_dropdown(models, key="liverun_model")
    if selected_model:
        st.code(f"Modelo selecionado: {selected_model['model_path']}")
        st.write(f"Framework: {selected_model['framework']}, Tipo: {selected_model['model_type']}")
        st.write(f"Treinado em: {selected_model.get('created_at', '')}")
        run_btn = st.button("Executar Previsão (Live Run)")
        if run_btn:
            st.info("Rodando live_run... (em breve: integração total)")
            # service.live_run(...)  # TODO: Adicionar chamada

