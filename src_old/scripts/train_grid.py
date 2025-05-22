
import os
import sys
src_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(src_root) if src_root not in sys.path else None
import pandas as pd
import numpy as np
import multiprocessing
import tensorflow as tf
import json

from models.db.model_binance import HourlyQuote
from data.data_processing import DataProcessor
from models.lstm_model import CustomLSTMTrainer
from optimization.grid_search import GridSearch
from utils.technical_indicators import TechnicalIndicators
from config.settings import *
from keras.callbacks import EarlyStopping
import random
import time
import hashlib



def model_trainer(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,model_path, **config):
    """
    Função que treina o modelo LSTM com base em 'config' e retorna as métricas (loss, val_loss).
    """

    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    trainer = CustomLSTMTrainer(
        input_shape=input_shape,
        layers_config=config.get('LAYERS_CONFIG', [64, 32]),
        dropout=config.get('DROPOUT', 0.2),
        batch_size=config.get('BATCH_SIZE', 16),
        epochs=config.get('EPOCHS', 50),
        patience=config.get('PATIENCE', 5),
        l1_reg= config.get('L1_REGULARIZATION', None),
        l2_reg= config.get('L2_REGULARIZATIONE', None),
        optimizer= config.get('OPTIMIZER', "Adam"),
        loss_fn= config.get('LOSS_FUNCTION', "mean_squared_error"),
        bidirectional= config.get('BIDIRECTIONAL', False),
        learning_rate= config.get('LEARNING_RATE', 0.001),
        recurrent_dropout= config.get('RECURRENT_DROPOUT', None),
        metrics= config.get('METRICS', None),
        activation_functions= config.get('ACTIVATION_FUNCTION', 'tanh'),
        output_units= config.get('OUTPUT_UNITS', 1),
    )

    lstm_model = trainer.train(
        X_train_scaled, y_train_scaled,
        X_val_scaled, y_val_scaled
    )
    history = lstm_model.history
    # Salvar o modelo
    trainer.save_model(lstm_model, model_path)

    # Obtemos as últimas métricas
    loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    return loss, val_loss


def run_combos_on_gpu(combos, gpu_index, end_date, relevant_cols, target_col, steps_ahead, output_csv):
    print(f"[Process GPU:{gpu_index}] Iniciando com {len(combos)} combinações...")

    with tf.device(f"/GPU:{gpu_index}"):
        data_df = HourlyQuote.get_to_date(end_date)
        data_df = TechnicalIndicators.process_indicators(data_df, Settings.INDICATORS_APPLY)
        data_df = data_df[relevant_cols]

        processor = DataProcessor(window_size=48)

        best_score = float("inf")
        best_config = None

        file_exists = os.path.exists(output_csv)

        for i, config in enumerate(combos, start=1):
            try:
                # Gerar seed única para esta configuração
                seed = random.randint(0, 2**32 - 1)
                set_seed(seed)

                if 'window_size' in config:
                    processor.window_size = config['window_size']
                
                X, y = processor.create_windows(data=data_df, coluna_alvo=target_col, steps_ahead=steps_ahead)
                X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)

                X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
                X_val_scaled, y_val_scaled = processor.apply_normalization(X_val, y_val)
                
                hash_input = json.dumps(config, sort_keys=True) + str(time.time())
                hash_input = hash_input.encode()
                model_hash = hashlib.md5(hash_input).hexdigest()[:8]
                model_name = f"model_gpu{gpu_index}_{model_hash}.h5"
                model_path = os.path.join("result/grid/models", model_name)

                # Obtemos o loss e val_loss
                loss, val_loss = model_trainer(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled,model_path, **config)

                if val_loss < best_score:
                    best_score = val_loss
                    best_config = config

                # Adicionar métricas ao registro
                row_dict = {
                    **config,
                    'train_loss': loss,
                    'val_loss': val_loss,
                    'best_so_far': val_loss == best_score,  # Flag para identificar melhor configuração
                    'seed': seed,
                    'model_path': model_path
                }
                df_temp = pd.DataFrame([row_dict])

                # Salvar no CSV
                df_temp.to_csv(
                    output_csv,
                    mode='a',
                    header=not file_exists,
                    index=False
                )
                file_exists = True

                print(f"[GPU:{gpu_index}] ({i}/{len(combos)}) - Config: {config} -> train_loss={loss:.5f}, val_loss={val_loss:.5f}")

            except Exception as e:
                print(f"[GPU:{gpu_index}] Erro na config {config}: {e}")

        print(f"[Process GPU:{gpu_index}] Finalizado. Melhor val_loss={best_score:.5f} com config={best_config}")
        print(f"[Process GPU:{gpu_index}] Resultados salvos continuamente em: {output_csv}")


def load_param_grid(file_path: str) -> dict:
    """
    Carrega o grid de parâmetros de um arquivo JSON.

    :param file_path: Caminho para o arquivo JSON.
    :return: Dicionário contendo os parâmetros.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo de grid de parâmetros não encontrado: {file_path}")
    
    with open(file_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    #### 1) Definimos um param_grid gigante
    param_grid_path = 'config/hyperparam_grid.json'
    param_grid = load_param_grid(param_grid_path)
    end_date = param_grid["END_DATE"]
    relevants_columns = param_grid["RELEVANT_COLUMNS"]
    target_column = param_grid["TARGET_COLUMN"]
    param_grid.pop("END_DATE", None)
    param_grid.pop("RELEVANT_COLUMNS", None)
    param_grid.pop("TARGET_COLUMN", None)
    #### 2) Usamos a classe GridSearch só para gerar TODAS as combinações
    gs = GridSearch(
        model_trainer=None,  # Temporário, não vamos chamar .search
        param_grid=param_grid,
        scoring='loss',
        verbose=1
    )
    all_combos = gs._generate_configurations()  # lista de dicionários
    print(f"Total de combinações geradas: {len(all_combos)}")


    #### 3) Dividimos a lista de combinações em duas metades
    half = len(all_combos)//2
    combos_half_1 = all_combos[:half]
    combos_half_2 = all_combos[half:]
    
    #### 4) Disparamos 2 processos, cada um rodando combos diferentes
    steps_ahead = 1

    # Nome dos CSVs de saída
    output_csv_1 = "result/grid/resultados_gpu0.csv"
    output_csv_2 = "result/grid/resultados_gpu1.csv"

    p1 = multiprocessing.Process(
        target=run_combos_on_gpu,
        args=(combos_half_1, 0, end_date, relevants_columns, target_column, steps_ahead, output_csv_1)
    )
    p2 = multiprocessing.Process(
        target=run_combos_on_gpu,
        args=(combos_half_2, 1, end_date, relevants_columns, target_column, steps_ahead, output_csv_2)
    )

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    print("\n======== PROCESSOS CONCLUÍDOS ========")
    print(f"Verifique os arquivos {output_csv_1} e {output_csv_2} para resultados.")
