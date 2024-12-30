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
from optimization.grid_search import GridSearch


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


def model_trainer(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, **config):
    """
    Função que treina o modelo LSTM com base em 'config' e retorna a métrica.
    
    :param X_train_scaled, y_train_scaled: Dados normalizados de treino
    :param X_val_scaled, y_val_scaled: Dados normalizados de validação
    :param config: Dicionário de hiperparâmetros (dropout, batch_size, epochs, etc.)
    :return: float -> valor da métrica (por exemplo, val_loss) ao final do treino
    """
    
    # Extraímos os hiperparâmetros do config
    dropout = config.get('dropout', 0.2)
    batch_size = config.get('batch_size', 16)
    epochs = config.get('epochs', 50)
    patience = config.get('patience', 5)
    layers_config = config.get('layers_config', [64, 32])
    
    # input_shape deve vir do shape de X_train_scaled
    # (samples, timesteps, features)
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])

    trainer = CustomLSTMTrainerGPU(
        input_shape=input_shape,
        layers_config=layers_config,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience
    )

    # Constrói o modelo
    model = trainer.build_model()

    # EarlyStopping para pegar o melhor modelo (você pode ajustar)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    # Treinamento (sem forçar GPU específica aqui — opcional)
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_val_scaled, y_val_scaled),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1  # silencioso
    )

    # Valor final de val_loss
    val_loss = history.history['val_loss'][-1]
    return val_loss


if __name__ == "__main__":
    data_df = load_data_from_db(END_DATE)
    if data_df.empty:
        raise ValueError("Nenhum dado foi recuperado do banco de dados.")

    # Mantemos apenas colunas relevantes
    data_df = data_df[RELEVANT_COLUMNS]

    # Cria instância do DataProcessorGPU
    processor = DataProcessorGPU(window_size=48)  # window_size default
    # Nesse grid search, iremos alterar window_size dentro do param_grid se quisermos

    # Define o param_grid para GridSearch
    param_grid = {
        "dropout": [0.2, 0.3, 0.4],
        "batch_size": [16, 32, 64, 128],
        "epochs": [50],
        "patience": [5],
        "layers_config": [
            [64, 32], [128, 64], [256, 128], [128, 64, 32], [256, 128, 64], [512, 256, 128],
            [64, 64], [128, 128], [256, 256], [128, 128, 128], [256, 256, 256], [512, 512, 512],
            [64, 64, 32, 32], [128, 128, 64, 64], [256, 256, 128, 128], [512, 512, 256, 256]
        ],
        "window_size": [48, 72, 96, 120]
    }


    # Instancia o GridSearch
    # Vamos usar 'loss' (val_loss) como métrica para minimizar
    grid = GridSearch(
        model_trainer=model_trainer,
        param_grid=param_grid,
        scoring='loss',
        verbose=2
    )

    # Executa a busca
    results = grid.search(
        data_processor=processor,
        data_df=data_df,
        coluna_alvo=TARGET_COLUMN,
        steps_ahead=STEPS_AHEAD
    )

    print("\n===== RESULTADOS DO GRID SEARCH =====")
    print("Melhores parâmetros:", results['best_params'])
    print("Melhor perda (loss):", results['best_score'])
    print("\nTabela de resultados:")
    print(results['results'])