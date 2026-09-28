import os
from dotenv import load_dotenv
import random
import numpy as np
import sys

load_dotenv()

# Caminho absoluto da raiz do projeto (onde está a pasta src)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Adiciona o diretório 'src' ao sys.path
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

# Caminho absoluto para a pasta 'config'
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_TYPE_CLASSNAME = {
    "lstm": "LSTM",
    "transformer": "Transformer"
}
FRAMEWORK_CLASSNAME = {
    "keras": "Keras",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
}

def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    except ImportError:
        pass

    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    print(f"Seed configurada para {seed} em todas as libs disponíveis.")

def set_cuda_tensorflow(gpu_index=0):
    """
    Seta a GPU visível no TensorFlow/Keras.
    Use ANTES de criar qualquer modelo!
    """
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            if gpu_index >= len(gpus):
                print(f"[WARN] GPU {gpu_index} não disponível. Usando GPU 0.")
                gpu_index = 0
            tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
            print(f"[INFO] TensorFlow usará a GPU: {gpus[gpu_index].name}")
        else:
            print("[INFO] Nenhuma GPU encontrada. Usando CPU.")
    except Exception as e:
        print(f"[ERRO] ao configurar GPU no TensorFlow: {e}")


def set_cuda_pytorch(gpu_index=0):
    """
    Seta a GPU no PyTorch.
    Retorna o device a ser usado ao criar o modelo (model.to(device)).
    """
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{gpu_index}")
            print(f"[INFO] PyTorch usará o device: {device}")
        else:
            device = torch.device("cpu")
            print("[INFO] CUDA não disponível. Usando CPU.")
        return device
    except Exception as e:
        print(f"[ERRO] ao configurar GPU no PyTorch: {e}")
        return torch.device("cpu")


class Settings:
    # Defaults para que importar o pacote nao dependa de um .env presente.
    # database/model_base.py constroi o MySQLDatabase em tempo de import e fazia
    # int(PORT_DB) com None, quebrando qualquer import sem .env — no CI, por exemplo.
    # peewee nao conecta na construcao, so quando a primeira consulta acontece.
    USER_DB = os.getenv("USER_DB", "root")
    PASSWORD_DB = os.getenv("PASSWORD_DB", "")
    HOST_DB = os.getenv("HOST_DB", "127.0.0.1")
    PORT_DB = os.getenv("PORT_DB", "3306")
    NAME_DB = os.getenv("NAME_DB", "nahas")
    
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

    USE_GPU = os.getenv("USE_GPU", "0").lower() in ("1", "true", "yes")
    SEED = int(os.getenv("SEED", "42"))
