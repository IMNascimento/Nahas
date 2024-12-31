from models.db.model_binance import HourlyQuote
from data.data_processing import DataProcessor
from models.lstm_model import CustomLSTMTrainer
from datetime import datetime
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

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
    """
    Carrega os dados do banco de dados até uma data específica.
    """
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
    """
    Salva os resultados em um arquivo CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Garante que o diretório existe
    results_df = pd.DataFrame({
        "timestamp": timestamps,
        "real_value": y_test.flatten(),
        "predicted_value": y_pred.flatten()
    })
    results_df.to_csv(output_path, index=False)
    print(f"Resultados salvos em: {output_path}")
    return results_df


def plot_results(results_df, save_path=None):
    """
    Plota os resultados previstos e reais e salva o gráfico se `save_path` for fornecido.
    """
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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Garante que o diretório existe
        plt.savefig(save_path, format='png')
        print(f"Gráfico salvo em: {save_path}")
    else:
        plt.show()

    plt.close()


def train_model(dropout, batch_size, epochs, patience, layers_config, window_size, csv_output_path, model_path, plot_output_path=None):
    """
    Função principal para treinar um modelo com os hiperparâmetros fornecidos.
    """
    print(f"Treinando modelo com DROPOUT={dropout}, BATCH_SIZE={batch_size}, EPOCHS={epochs}, LAYERS={layers_config}")
    data_df = HourlyQuote.get_to_date(END_DATE)
    if data_df.empty:
        raise ValueError("Nenhum dado foi recuperado do banco de dados. Verifique a data ou os dados disponíveis.")
    original_timestamps = data_df['timestamp'].values
    data_df = data_df[RELEVANT_COLUMNS]
    processor = DataProcessor(window_size=window_size)
    X, y = processor.create_windows(data=data_df, coluna_alvo=TARGET_COLUMN, steps_ahead=STEPS_AHEAD)
    X_train, X_validation, X_test, y_train, y_validation, y_test = processor.split_data(X, y)

    # Normalização
    X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
    X_validation_scaled, y_validation_scaled = processor.apply_normalization(X_validation, y_validation)
    X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)

    # Configuração do modelo
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    trainer = CustomLSTMTrainer(
        input_shape=input_shape,
        layers_config=layers_config,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience
    )

    # Treina o modelo
    lstm_model = trainer.train(X_train_scaled, y_train_scaled, X_validation_scaled, y_validation_scaled)
    trainer.save_model(lstm_model, model_path=model_path)

    # Previsão
    y_pred_scaled = lstm_model.predict(X_test_scaled)
    processor.load_scaler()
    y_pred = processor.inverse_transform(y_pred_scaled).flatten()
    y_test = processor.inverse_transform(y_test_scaled).flatten()
    timestamps = original_timestamps[-len(y_test):]

    # Verifica tamanhos e salva os resultados
    assert len(timestamps) == len(y_test) == len(y_pred), \
        f"Dimensões incompatíveis: timestamps({len(timestamps)}), y_test({len(y_test)}), y_pred({len(y_pred)})"
    results_df = save_results_to_csv(timestamps, y_test, y_pred, csv_output_path)
    plot_results(results_df, save_path=plot_output_path)

    print(f"Modelo salvo em: {model_path}")
    print(f"Resultados salvos em: {csv_output_path}")


if __name__ == "__main__":
    # Configurações para diferentes modelos
    configurations = [
    {"dropout": 0.2, "batch_size": 16,"epochs": 50, "layers_config": [64, 32], "window_size": 48,"patience": 5},
    {"dropout": 0.2, "batch_size": 16,"epochs": 50, "layers_config": [64, 32], "window_size": 72,"patience": 5},
    {"dropout": 0.2, "batch_size": 16,"epochs": 50, "layers_config": [64, 32], "window_size": 96,"patience": 5},
    {"dropout": 0.2, "batch_size": 32,"epochs": 50, "layers_config": [64, 64], "window_size": 48,"patience": 5},
    {"dropout": 0.2, "batch_size": 32,"epochs": 50, "layers_config": [64, 64], "window_size": 72,"patience": 5},
    {"dropout": 0.2, "batch_size": 32,"epochs": 50, "layers_config": [64, 64], "window_size": 96,"patience": 5},
    {"dropout": 0.2, "batch_size": 64,"epochs": 50, "layers_config": [128, 64], "window_size": 48,"patience": 5},
    {"dropout": 0.2, "batch_size": 64,"epochs": 50, "layers_config": [128, 64], "window_size": 72,"patience": 5},
    {"dropout": 0.2, "batch_size": 64,"epochs": 50, "layers_config": [128, 64], "window_size": 96,"patience": 5},
    {"dropout": 0.3, "batch_size": 16,"epochs": 50, "layers_config": [128, 128], "window_size": 48,"patience": 5},
    {"dropout": 0.3, "batch_size": 16,"epochs": 50, "layers_config": [128, 128], "window_size": 72,"patience": 5},
    {"dropout": 0.3, "batch_size": 16,"epochs": 50, "layers_config": [128, 128], "window_size": 96,"patience": 5},
    {"dropout": 0.3, "batch_size": 32,"epochs": 50, "layers_config": [256, 128], "window_size": 48,"patience": 5},
    {"dropout": 0.3, "batch_size": 32,"epochs": 50, "layers_config": [256, 128], "window_size": 72,"patience": 5},
    {"dropout": 0.3, "batch_size": 32,"epochs": 50, "layers_config": [256, 128], "window_size": 96,"patience": 5},
    {"dropout": 0.3, "batch_size": 64,"epochs": 50, "layers_config": [256, 128], "window_size": 48,"patience": 5},
    {"dropout": 0.3, "batch_size": 64,"epochs": 50, "layers_config": [256, 128], "window_size": 72,"patience": 5},
    {"dropout": 0.3, "batch_size": 64,"epochs": 50, "layers_config": [256, 128], "window_size": 96,"patience": 5},
    {"dropout": 0.4, "batch_size": 16,"epochs": 50, "layers_config": [64, 64, 32], "window_size": 48,"patience": 5},
    {"dropout": 0.4, "batch_size": 16,"epochs": 50, "layers_config": [64, 64, 32], "window_size": 72,"patience": 5},
    {"dropout": 0.4, "batch_size": 16,"epochs": 50, "layers_config": [64, 64, 32], "window_size": 96,"patience": 5},
    {"dropout": 0.4, "batch_size": 32,"epochs": 50, "layers_config": [128, 64, 32], "window_size": 48,"patience": 5},
    {"dropout": 0.4, "batch_size": 32,"epochs": 50, "layers_config": [128, 64, 32], "window_size": 72,"patience": 5},
    {"dropout": 0.4, "batch_size": 32,"epochs": 50, "layers_config": [128, 64, 32], "window_size": 96,"patience": 5},
    {"dropout": 0.4, "batch_size": 64,"epochs": 50, "layers_config": [128, 128, 64], "window_size": 48,"patience": 5},
    {"dropout": 0.4, "batch_size": 64,"epochs": 50, "layers_config": [128, 128, 64], "window_size": 72,"patience": 5},
    {"dropout": 0.4, "batch_size": 64,"epochs": 50, "layers_config": [128, 128, 64], "window_size": 96,"patience": 5},
    {"dropout": 0.2, "batch_size": 16,"epochs": 50, "layers_config": [256, 128], "window_size": 48,"patience": 5},
    {"dropout": 0.2, "batch_size": 16,"epochs": 50, "layers_config": [256, 128], "window_size": 72,"patience": 5},
    {"dropout": 0.2, "batch_size": 16,"epochs": 50, "layers_config": [256, 128], "window_size": 96,"patience": 5},
    {"dropout": 0.2, "batch_size": 32,"epochs": 50, "layers_config": [256, 128], "window_size": 48,"patience": 5},
    {"dropout": 0.2, "batch_size": 32,"epochs": 50, "layers_config": [256, 128], "window_size": 72,"patience": 5},
    {"dropout": 0.2, "batch_size": 32,"epochs": 50, "layers_config": [256, 128], "window_size": 96,"patience": 5},
    {"dropout": 0.3, "batch_size": 16,"epochs": 50, "layers_config": [128, 64, 32], "window_size": 48,"patience": 5},
    {"dropout": 0.3, "batch_size": 16,"epochs": 50, "layers_config": [128, 64, 32], "window_size": 72,"patience": 5},
    {"dropout": 0.3, "batch_size": 16,"epochs": 50, "layers_config": [128, 64, 32], "window_size": 96,"patience": 5},
    {"dropout": 0.3, "batch_size": 32,"epochs": 50, "layers_config": [64, 64], "window_size": 48,"patience": 5},
    {"dropout": 0.3, "batch_size": 32,"epochs": 50, "layers_config": [64, 64], "window_size": 72,"patience": 5},
    {"dropout": 0.3, "batch_size": 32,"epochs": 50, "layers_config": [64, 64], "window_size": 96,"patience": 5},
    {"dropout": 0.4, "batch_size": 16,"epochs": 50, "layers_config": [128, 128], "window_size": 48,"patience": 5},
    {"dropout": 0.4, "batch_size": 16,"epochs": 50, "layers_config": [128, 128], "window_size": 72,"patience": 5},
    {"dropout": 0.4, "batch_size": 16,"epochs": 50, "layers_config": [128, 128], "window_size": 96,"patience": 5},
    {"dropout": 0.4, "batch_size": 64,"epochs": 50, "layers_config": [64, 64, 32], "window_size": 48,"patience": 5},
    {"dropout": 0.4, "batch_size": 64,"epochs": 50, "layers_config": [64, 64, 32], "window_size": 72,"patience": 5},
    {"dropout": 0.4, "batch_size": 64,"epochs": 50, "layers_config": [64, 64, 32], "window_size": 96,"patience": 5},
]

    # Loop para treinar diferentes modelos
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