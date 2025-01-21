import os
from dotenv import load_dotenv
import json
import random
import numpy as np
import tensorflow as tf

# Diretório raiz do projeto (src)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))  # Caminho até `src/config/settings.py`
SRC_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))
# Caminho para o arquivo JSON de hiperparâmetros
HYPERPARAMETERS_FILE = os.path.join(SRC_ROOT, "config", "hyperparameters.json")

# Carregar o arquivo .env
load_dotenv()

def set_seed(seed: int):
    """
    Configura a seed para garantir reprodutibilidade em todas as operações,
    incluindo GPU e CPU, além das bibliotecas usadas pelo TensorFlow, NumPy,
    e Python.

    :param seed: O valor da seed a ser configurado.
    """
    # Seed para hash interno do Python (reprodutibilidade em estruturas de dados)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Seed para bibliotecas Python que utilizam números aleatórios
    random.seed(seed)
    
    # Seed para NumPy
    np.random.seed(seed)
    
    # Seed para TensorFlow (abrange CPU e GPU)
    tf.random.set_seed(seed)
    
    # Garantir determinismo em TensorFlow
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Operações determinísticas
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # Determinismo para cuDNN

    # Log da seed configurada
    print(f"Seed configurada para {seed} em todas as bibliotecas.")

class Settings:
    USER_DB = os.getenv("USER_DB")
    PASSWORD_DB = os.getenv("PASSWORD_DB")
    HOST_DB = os.getenv("HOST_DB")
    PORT_DB = os.getenv("PORT_DB")
    NAME_DB = os.getenv("NAME_DB")

    EMAIL_SMTP = os.getenv("EMAIL_SMTP")
    EMAIL_PORT = os.getenv("EMAIL_PORT")
    EMAIL_USER = os.getenv("EMAIL_USER")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    EMAIL_IMAP = os.getenv("EMAIL_IMAP")

    USER_MT5 = os.getenv("LOGIN_MT5")
    PASSWORD_MT5 = os.getenv("PASSWORD_MT5")
    SERVER_MT5 = os.getenv("SERVER_MT5")

    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
    BINANCE_API_SECRET_KEY = os.getenv("BINANCE_API_SECRET_KEY")

    USE_GPU = os.getenv("USE_GPU")
    SEED = int(os.getenv("SEED"))

    
    with open(HYPERPARAMETERS_FILE, 'r') as f:
        hyperparameters = json.load(f)

    # Hiperparâmetros gerais
    END_DATE = hyperparameters.get("END_DATE")
    START_DATE = hyperparameters.get("START_DATE")
    WINDOW_SIZE = hyperparameters.get("WINDOW_SIZE",48)
    STEPS_AHEAD = hyperparameters.get("STEPS_AHEAD",1)
    LAYERS_CONFIG = hyperparameters.get("LAYERS_CONFIG",[64,32])
    ACTIVATION_FUNCTION = hyperparameters.get("ACTIVATION_FUNCTION",["tanh","tanh"])
    BIDIRECTIONAL = hyperparameters.get("BIDIRECTIONAL",False)
    DROPOUT = hyperparameters.get("DROPOUT",0.3)
    RECURRENT_DROPOUT = hyperparameters.get("RECURRENT_DROPOUT", None)
    L1_REGULARIZATION = hyperparameters.get("L1_REGULARIZATION", None)
    L2_REGULARIZATION = hyperparameters.get("L2_REGULARIZATION", None)
    LEARNING_RATE = hyperparameters.get("LEARNING_RATE", 0.0001)
    OUTPUT_UNITS = hyperparameters.get("OUTPUT_UNITS",1)
    OPTIMIZER = hyperparameters.get("OPTIMIZER", "Adam")
    BATCH_SIZE = hyperparameters.get("BATCH_SIZE", 32)
    EPOCHS = hyperparameters.get("EPOCHS", 50)
    PATIENCE = hyperparameters.get("PATIENCE", 5)
    LOSS_FUNCTION = hyperparameters.get("LOSS_FUNCTION", "mean_squared_error")
    INDICATORS_APPLY = hyperparameters.get("INDICATORS_APPLY", {})
    RELEVANT_COLUMNS = hyperparameters.get("RELEVANT_COLUMNS", ['close', 'open', 'low', 'high', 'volume'])
    TARGET_COLUMN = hyperparameters.get("TARGET_COLUMN", "close")
    METRICS = hyperparameters.get("METRICS", [])
    VALIDATION_SPLIT = hyperparameters.get("VALIDATION_SPLIT", 0.15)
    TRAIN_SIZE= hyperparameters.get("TRAIN_SIZE", 0.70)
    LOAD_MODEL= hyperparameters.get("LOAD_MODEL")
    LOAD_MODEL_FINETUNING= hyperparameters.get("LOAD_MODEL_FINETUNING")