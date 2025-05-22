import os
import sys
import json
import time
import random
import hashlib
import multiprocessing
import pandas as pd
from datetime import datetime

# Ajuste caminho se necessário
src_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(src_root) if src_root not in sys.path else None

from data.data_processing import DataProcessor
from database.model_binance import HourlyQuoteBitcoin
from utils.technical_indicators import TechnicalIndicators
from config.settings import set_seed, Settings
from optimization.grid_search import GridSearch

import importlib

def get_trainer_class(framework: str, model_type: str):
    """
    Importa e retorna a classe trainer correta conforme o framework e model_type.
    """
    module_path = f"models.{framework.lower()}.{model_type.lower()}_trainer"
    class_name = f"{framework.capitalize()}{model_type.capitalize()}Trainer"
    try:
        module = importlib.import_module(module_path)
        trainer_class = getattr(module, class_name)
        return trainer_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Trainer não encontrado: {module_path}.{class_name}\nErro: {e}")

def universal_model_trainer(X_train, y_train, X_val, y_val, model_path, framework, model_type, **config):
    """
    Função universal que treina qualquer modelo (Keras, PyTorch, TensorFlow, LSTM, Transformer).
    """
    input_shape = (X_train.shape[1], X_train.shape[2])
    trainer_class = get_trainer_class(framework, model_type)
    # Remove argumentos que não são hiperparâmetros do trainer
    trainer_kwargs = {k: v for k, v in config.items() if k not in ["framework", "model_type"]}
    trainer = trainer_class(input_shape=input_shape, **trainer_kwargs)
    model = trainer.train(X_train, y_train, X_val, y_val)
    trainer.save_model(model, model_path)
    # Acessa as métricas
    if hasattr(model, "history"):
        history = model.history
        loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
    elif hasattr(trainer, "get_last_metrics"):
        loss, val_loss = trainer.get_last_metrics()
    else:
        loss = val_loss = None
    return loss, val_loss

def run_combos_on_device(combos, device_index, end_date, relevant_cols, target_col, steps_ahead, output_csv, framework, model_type):
    print(f"[Process {device_index}] Iniciando com {len(combos)} combinações... (Framework={framework}, Modelo={model_type})")
    if framework.lower() == "keras" or framework.lower() == "tensorflow":
        import tensorflow as tf
        device_str = f"/GPU:{device_index}" if tf.config.list_physical_devices('GPU') else "/CPU:0"
        device_ctx = tf.device(device_str)
    else:
        device_ctx = None  # PyTorch geralmente gerencia internamente

    if device_ctx:
        device_ctx.__enter__()

    data_df = HourlyQuoteBitcoin.get_to_date(end_date)
    data_df = TechnicalIndicators.process_indicators(data_df, Settings.INDICATORS_APPLY)
    data_df = data_df[relevant_cols]

    best_score = float("inf")
    best_config = None
    file_exists = os.path.exists(output_csv)

    for i, config in enumerate(combos, start=1):
        try:
            # Seed única
            seed = random.randint(0, 2**32 - 1)
            set_seed(seed)

            window_size = config.get('window_size') or config.get('WINDOW_SIZE') or Settings.WINDOW_SIZE
            processor = DataProcessor(window_size=window_size)

            X, y = processor.create_windows(data=data_df, coluna_alvo=target_col, steps_ahead=steps_ahead)
            X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)
            X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
            X_val_scaled, y_val_scaled = processor.apply_normalization(X_val, y_val)

            hash_input = json.dumps(config, sort_keys=True) + str(time.time())
            model_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            ext = "h5" if framework.lower() == "keras" else "pt" if framework.lower() == "pytorch" else "model"
            model_name = f"model_{framework.lower()}_{model_type.lower()}_{device_index}_{model_hash}.{ext}"
            model_path = os.path.join("result/grid/models", model_name)

            # Framework/model universal trainer
            loss, val_loss = universal_model_trainer(
                X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,
                model_path, framework, model_type, **config
            )

            if val_loss is not None and val_loss < best_score:
                best_score = val_loss
                best_config = config

            # Registrar resultado
            row_dict = {
                **config,
                'framework': framework,
                'model_type': model_type,
                'train_loss': loss,
                'val_loss': val_loss,
                'best_so_far': val_loss == best_score,
                'seed': seed,
                'model_path': model_path
            }
            df_temp = pd.DataFrame([row_dict])
            df_temp.to_csv(
                output_csv,
                mode='a',
                header=not file_exists,
                index=False
            )
            file_exists = True

            print(f"[{framework}][{model_type}] ({i}/{len(combos)}) - Config: {config} -> train_loss={loss}, val_loss={val_loss}")

        except Exception as e:
            print(f"[{framework}][{model_type}] Erro na config {config}: {e}")

    if device_ctx:
        device_ctx.__exit__(None, None, None)

    print(f"[Process {device_index}] Finalizado. Melhor val_loss={best_score} com config={best_config}")
    print(f"[Process {device_index}] Resultados salvos continuamente em: {output_csv}")

def load_param_grid(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo de grid de parâmetros não encontrado: {file_path}")
    with open(file_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    # Carregar param_grid do arquivo JSON/YAML
    param_grid_path = 'config/hyperparam_grid.json'
    param_grid = load_param_grid(param_grid_path)
    end_date = param_grid["END_DATE"]
    relevants_columns = param_grid["RELEVANT_COLUMNS"]
    target_column = param_grid["TARGET_COLUMN"]
    framework = param_grid.get("FRAMEWORK", "keras")  # Adicione "FRAMEWORK" ao seu grid json!
    model_type = param_grid.get("MODEL_TYPE", "lstm") # Adicione "MODEL_TYPE" ao seu grid json!

    # Remova chaves que não são hiperparâmetros
    for key in ["END_DATE", "RELEVANT_COLUMNS", "TARGET_COLUMN", "FRAMEWORK", "MODEL_TYPE"]:
        param_grid.pop(key, None)

    # Gerar combinações
    gs = GridSearch(
        model_trainer=None,  # só usamos ._generate_configurations()
        param_grid=param_grid,
        scoring='loss',
        verbose=1
    )
    all_combos = gs._generate_configurations()
    print(f"Total de combinações geradas: {len(all_combos)}")

    # Divida combos entre os dispositivos
    n_gpus = 2  # ou detecte automaticamente se quiser
    chunk_size = len(all_combos) // n_gpus
    combos_split = [all_combos[i*chunk_size:(i+1)*chunk_size] for i in range(n_gpus)]
    output_csvs = [f"result/grid/resultados_{framework}_{model_type}_gpu{i}.csv" for i in range(n_gpus)]

    # Dispare os processos
    processes = []
    for i in range(n_gpus):
        p = multiprocessing.Process(
            target=run_combos_on_device,
            args=(combos_split[i], i, end_date, relevants_columns, target_column, 1, output_csvs[i], framework, model_type)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\n======== PROCESSOS CONCLUÍDOS ========")
    for path in output_csvs:
        print(f"Verifique os arquivos {path} para resultados.")
