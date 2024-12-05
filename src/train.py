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
    results_df = pd.DataFrame({
        "timestamp": timestamps,
        "real_value": y_test.flatten(),
        "predicted_value": y_pred.flatten()
    })
    results_df.to_csv(output_path, index=False)
    print(f"Resultados salvos em: {output_path}")
    return results_df


def plot_results(results_df):
    """
    Plota os resultados previstos e reais.
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
    plt.show()




def main():
    data_df = load_data_from_db(END_DATE)
    if data_df.empty:
        raise ValueError("Nenhum dado foi recuperado do banco de dados. Verifique a data ou os dados disponíveis.")
    original_timestamps = data_df['timestamp'].values
    data_df = data_df[RELEVANT_COLUMNS]
    processor = DataProcessor(window_size=WINDOW_SIZE)
    X, y = processor.create_windows(data=data_df, coluna_alvo=TARGET_COLUMN, steps_ahead=STEPS_AHEAD)
    print(f"TREINO: Dimensões de X e y antes da normalização: X={X.shape}, y={y.shape}")
    X_train, X_validation, X_test, y_train, y_validation, y_test = processor.split_data(X, y)
    
    X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
    X_validation_scaled, y_validation_scaled = processor.apply_normalization(X_validation, y_validation)
    X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)
    print(f"TREINO: Dimensões de X_train_scaled: {X_train_scaled.shape}, y_train_scaled: {y_train_scaled.shape}")
    print(f"TREINO: Dimensões de X_validation_scaled: {X_validation_scaled.shape}, y_validation_scaled: {y_validation_scaled.shape}")
    print(f"TREINO: Dimensões de X_test_scaled: {X_test_scaled.shape}, y_test_scaled: {y_test_scaled.shape}")
    processor.save_scaler()

    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    trainer = CustomLSTMTrainer(
        input_shape=input_shape,
        layers_config=LAYERS_CONFIG,
        dropout=DROPOUT,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        patience=PATIENCE
    )
    lstm_model = trainer.train(X_train_scaled, y_train_scaled, X_validation_scaled, y_validation_scaled)
    trainer.save_model(lstm_model, model_path=MODEL_PATH)

    y_pred_scaled = lstm_model.predict(X_test_scaled)
    processor.load_scaler()
    print(f"TREINO: Dimensões de y_pred_scaled: {y_pred_scaled.shape}")
    y_pred = processor.inverse_transform(y_pred_scaled).flatten()
    y_test = processor.inverse_transform(y_test_scaled).flatten()
    print(f"TREINO: Dimensões de y_pred após inverse_transform: {y_pred.shape}")
    print(f"TREINO: Dimensões de y_test após inverse_transform: {y_test.shape}")
    timestamps = original_timestamps[-len(y_test):]
    # Verifica se todos os arrays têm o mesmo tamanho
    assert len(timestamps) == len(y_test) == len(y_pred), \
    f"Dimensões incompatíveis: timestamps({len(timestamps)}), y_test({len(y_test)}), y_pred({len(y_pred)})"
    results_df = pd.DataFrame({
        "timestamp": timestamps,
        "real_value": y_test,
        "predicted_value": y_pred
    })
    print(f"Tamanhos: timestamps({len(timestamps)}), y_test_flat({len(y_test.flatten())}), y_pred_flat({len(y_pred.flatten())})")
    save_results_to_csv(timestamps, y_test, y_pred, CSV_OUTPUT_PATH)
    plot_results(results_df)

    print(f"Resultados salvos em: {CSV_OUTPUT_PATH}")


if __name__ == "__main__":
    main()