
import os
import pandas as pd
import numpy as np
import multiprocessing
import tensorflow as tf

from models.db.model_binance import HourlyQuote
from data.data_gpu_processing import DataProcessorGPU
from models.lstm_gpu import CustomLSTMTrainerGPU
from optimization.grid_search import GridSearch
from keras.callbacks import EarlyStopping


def load_data_from_db(end_date):
    query = (HourlyQuote
             .select()
             .where(HourlyQuote.timestamp <= end_date)
             .order_by(HourlyQuote.timestamp))
    data = pd.DataFrame(list(query.dicts()))
    if not data.empty:
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        return data
    else:
        print("Nenhum dado encontrado no banco.")
        return pd.DataFrame()


def model_trainer(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, **config):
    """
    Função que treina o modelo LSTM com base em 'config' e retorna a métrica (val_loss).
    """
    dropout = config.get('dropout', 0.2)
    batch_size = config.get('batch_size', 16)
    epochs = config.get('epochs', 50)
    patience = config.get('patience', 5)
    layers_config = config.get('layers_config', [64, 32])

    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    trainer = CustomLSTMTrainerGPU(
        input_shape=input_shape,
        layers_config=layers_config,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience
    )
    model = trainer.build_model()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    val_loss = history.history['val_loss'][-1]
    return val_loss


def run_combos_on_gpu(combos, gpu_index, end_date, relevant_cols, target_col, steps_ahead, output_csv):
    """
    Roda uma 'mini' busca manual em 'combos' (lista de configs), usando a GPU especificada.
    Salva os resultados em 'output_csv'.
    """
    print(f"[Process GPU:{gpu_index}] Iniciando com {len(combos)} combinações...")

    # Força uso de uma GPU específica
    with tf.device(f"/GPU:{gpu_index}"):
        # Carrega dados
        data_df = load_data_from_db(end_date)
        data_df = data_df[relevant_cols]

        # Instancia o DataProcessor
        processor = DataProcessorGPU(window_size=48)  # default; combos podem alterar window_size

        # Variáveis para rastrear melhor score
        best_score = float("inf")
        best_config = None
        results = []

        for i, config in enumerate(combos, start=1):
            try:
                # Atualiza window_size (se estiver no config)
                if 'window_size' in config:
                    processor.window_size = config['window_size']
                
                # Cria janelas
                X, y = processor.create_windows(data=data_df, coluna_alvo=target_col, steps_ahead=steps_ahead)
                X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(X, y)

                # Normaliza
                X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
                X_val_scaled, y_val_scaled = processor.apply_normalization(X_val, y_val)
                # (Não precisamos necessariamente do X_test aqui se só queremos val_loss)

                # Treina e obtém val_loss
                val_loss = model_trainer(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, **config)

                if val_loss < best_score:
                    best_score = val_loss
                    best_config = config

                results.append({**config, 'loss': val_loss})
                print(f"[GPU:{gpu_index}] ({i}/{len(combos)}) - Config: {config} -> loss={val_loss:.5f}")

            except Exception as e:
                print(f"[GPU:{gpu_index}] Erro na config {config}: {e}")

        # Salva CSV
        import pandas as pd
        df_res = pd.DataFrame(results)
        df_res.to_csv(output_csv, index=False)
        print(f"[Process GPU:{gpu_index}] Finalizado. Melhor loss={best_score:.5f} com config={best_config}")
        print(f"[Process GPU:{gpu_index}] Resultados salvos em: {output_csv}")


if __name__ == "__main__":

    #### 1) Definimos um param_grid gigante
    param_grid = {
        "dropout": [0.2, 0.3, 0.4],
        "batch_size": [16, 32, 64, 128],
        "epochs": [50],
        "patience": [5],
        "layers_config": [
            [64, 32], [128, 64], [256, 128],
            [128, 64, 32], [256, 128, 64], [512, 256, 128],
            [64, 64], [128, 128], [256, 256],
            [128, 128, 128], [256, 256, 256], [512, 512, 512],
            [64, 64, 32, 32], [128, 128, 64, 64],
            [256, 256, 128, 128], [512, 512, 256, 256]
        ],
        "window_size": [48, 72, 96, 120]
    }

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
    end_date = "2024-09-14 23:59:59"
    relevant_cols = ["open", "high", "low", "close", "volume"]
    target_col = "close"
    steps_ahead = 1

    # Nome dos CSVs de saída
    output_csv_1 = "resultados_gpu0.csv"
    output_csv_2 = "resultados_gpu1.csv"

    p1 = multiprocessing.Process(
        target=run_combos_on_gpu,
        args=(combos_half_1, 0, end_date, relevant_cols, target_col, steps_ahead, output_csv_1)
    )
    p2 = multiprocessing.Process(
        target=run_combos_on_gpu,
        args=(combos_half_2, 1, end_date, relevant_cols, target_col, steps_ahead, output_csv_2)
    )

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    print("\n======== PROCESSOS CONCLUÍDOS ========")
    print(f"Verifique os arquivos {output_csv_1} e {output_csv_2} para resultados.")