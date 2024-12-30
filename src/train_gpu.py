# train_gpu.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime
from keras.callbacks import EarlyStopping

from models.db.model_binance import HourlyQuote
from data.data_gpu_processing import DataProcessorGPU
from models.lstm_gpu import CustomLSTMTrainerGPU

# Configurações
WINDOW_SIZE = 48
STEPS_AHEAD = 1
M_PATH = "result/models/"
END_DATE = "2024-09-14 23:59:59"
RELEVANT_COLUMNS = ["open", "high", "low", "close", "volume"]
TARGET_COLUMN = "close"
DROPOUT = 0.2
BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 5
LAYERS_CONFIG = [64, 32]
CSV_OUTPUT_PATH = f"treino_{DROPOUT}_{BATCH_SIZE}_{EPOCHS}_{PATIENCE}_{LAYERS_CONFIG}_{WINDOW_SIZE}.csv"
NAME_MODEL = f"lstm_model_{DROPOUT}_{BATCH_SIZE}_{EPOCHS}_{PATIENCE}_{LAYERS_CONFIG}_{WINDOW_SIZE}.h5"
MODEL_PATH = os.path.join(M_PATH, NAME_MODEL)

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

def save_results_to_csv(timestamps, y_test, y_pred, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df = pd.DataFrame({
        "timestamp": timestamps,
        "real_value": y_test.flatten(),
        "predicted_value": y_pred.flatten()
    })
    results_df.to_csv(output_path, index=False)
    print(f"Resultados salvos em: {output_path}")
    return results_df

def plot_results(results_df, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(results_df["timestamp"], results_df["real_value"], label="Real")
    plt.plot(results_df["timestamp"], results_df["predicted_value"], label="Previsto")
    plt.legend()
    plt.title("Previsão de Preço com LSTM")
    plt.xlabel("Timestamp")
    plt.ylabel("Preço")
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, format='png')
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()

    plt.close()

def train_model(
    dropout, batch_size, epochs, patience, layers_config,
    window_size, csv_output_path, model_path, plot_output_path=None
):
    print(f"Treinando modelo com DROPOUT={dropout}, BATCH_SIZE={batch_size}, EPOCHS={epochs}, LAYERS={layers_config}")

    data_df = load_data_from_db(END_DATE)
    if data_df.empty:
        raise ValueError("Nenhum dado foi recuperado do banco de dados. Verifique a data ou os dados disponíveis.")

    original_timestamps = data_df['timestamp'].values
    data_df = data_df[RELEVANT_COLUMNS]

    processor = DataProcessorGPU(window_size=window_size)
    X, y = processor.create_windows(
        data=data_df,
        coluna_alvo=TARGET_COLUMN,
        steps_ahead=STEPS_AHEAD
    )
    X_train, X_validation, X_test, y_train, y_validation, y_test = processor.split_data(X, y)

    # Normalização
    X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
    X_validation_scaled, y_validation_scaled = processor.apply_normalization(X_validation, y_validation)
    X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)

    # Configuração do modelo
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    trainer = CustomLSTMTrainerGPU(
        input_shape=input_shape,
        layers_config=layers_config,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience
    )

    # Converte tudo para float32 antes de treinar (caso seja NumPy)
    X_train_scaled = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
    y_train_scaled = tf.convert_to_tensor(y_train_scaled, dtype=tf.float32)
    X_validation_scaled = tf.convert_to_tensor(X_validation_scaled, dtype=tf.float32)
    y_validation_scaled = tf.convert_to_tensor(y_validation_scaled, dtype=tf.float32)

    # Força o uso de apenas 1 GPU, se disponível
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        device_to_use = "/GPU:0"
        print("Treinando no dispositivo:", device_to_use)
    else:
        device_to_use = "/CPU:0"
        print("Nenhuma GPU encontrada. Treinando em CPU.")

    with tf.device(device_to_use):
        # Treina o modelo
        lstm_model = trainer.train(
            X_train_scaled, y_train_scaled,
            X_validation_scaled, y_validation_scaled
        )

    # Salva o modelo
    trainer.save_model(lstm_model, model_path=model_path)

    # Previsão
    X_test_scaled = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
    y_pred_scaled = lstm_model.predict(X_test_scaled)

    # Carrega scalers e inverte normalização
    processor.load_scaler()
    y_pred = processor.inverse_transform(y_pred_scaled.numpy()).flatten()
    y_test = processor.inverse_transform(y_test_scaled).flatten()
    timestamps = original_timestamps[-len(y_test):]

    assert len(timestamps) == len(y_test) == len(y_pred), (
        f"Dimensões incompatíveis: timestamps({len(timestamps)}), "
        f"y_test({len(y_test)}), y_pred({len(y_pred)})"
    )

    results_df = save_results_to_csv(timestamps, y_test, y_pred, csv_output_path)
    plot_results(results_df, save_path=plot_output_path)

    print(f"Modelo salvo em: {model_path}")
    print(f"Resultados salvos em: {csv_output_path}")





if __name__ == "__main__":
    configurations = [
        {"dropout": 0.2, "batch_size": 16,"epochs": 50, "layers_config": [64, 32], "window_size": 48,"patience": 5},
        {"dropout": 0.2, "batch_size": 16,"epochs": 50, "layers_config": [64, 32], "window_size": 72,"patience": 5},
        # ...
        # Remova ou adicione quantas configurações quiser
    ]

    for config in configurations:
        csv_output_path = f"result/csv/treinos/treino_dropout_{config['dropout']}_batch_{config['batch_size']}_window_{config['window_size']}_layers_{config['layers_config']}.csv"
        model_path = f"result/models/model_dropout_{config['dropout']}_batch_{config['batch_size']}_window_{config['window_size']}_layers_{config['layers_config']}.h5"
        plot_output_path = f"result/graficos/treinos/model_dropout_{config['dropout']}_batch_{config['batch_size']}_window_{config['window_size']}_layers_{config['layers_config']}.png"
        
        try:
            train_model(
                dropout=config["dropout"],
                batch_size=config["batch_size"],
                epochs=config["epochs"],
                patience=config["patience"],
                layers_config=config["layers_config"],
                window_size=config["window_size"],
                csv_output_path=csv_output_path,
                model_path=model_path,
                plot_output_path=plot_output_path
            )
        except Exception as e:
            print(f"Erro ao treinar o modelo com configuração {config}: {e}")