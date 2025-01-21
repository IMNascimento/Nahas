import os
import sys
src_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(src_root) if src_root not in sys.path else None
from datetime import datetime
import time
import argparse
import tensorflow as tf
from keras.models import load_model
from keras.callbacks import EarlyStopping
from config.settings import *
from utils.technical_indicators import TechnicalIndicators
from models.db.model_binance import HourlyQuote
from data.data_processing import DataProcessor
from utils.plotter import Plotter
from utils.csv_exporter import CSVExporter

# Configura o caminho raiz
src_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(src_root) if src_root not in sys.path else None

def parse_args():
    """
    Define os argumentos que podem ser passados via terminal.
    """
    parser = argparse.ArgumentParser(description="Fine-tuning de um modelo LSTM existente.")
    
    parser.add_argument("--model_path", type=str, default=Settings.LOAD_MODEL, help="Caminho do modelo salvo para fine-tuning.")
    parser.add_argument("--window_size", type=int, default=Settings.WINDOW_SIZE, help="Tamanho da janela temporal.")
    parser.add_argument("--epochs", type=int, default=Settings.EPOCHS, help="Número de épocas para fine-tuning.")
    parser.add_argument("--batch_size", type=int, default=Settings.BATCH_SIZE, help="Tamanho do batch.")
    parser.add_argument("--patience", type=int, default=Settings.PATIENCE, help="Número de épocas para early stopping.")
    parser.add_argument("--learning_rate", type=float, default=Settings.LEARNING_RATE, help="Taxa de aprendizado para fine-tuning.")

    return parser.parse_args()

def create_finetuning_folder():
    """
    Cria uma pasta exclusiva para armazenar os resultados do fine-tuning.
    A estrutura será:
    result/
        fine_tuning/
            hash_model/
                data_hora/
    """
    # Extrai o hash do modelo do nome do arquivo carregado
    model_name = os.path.basename(Settings.LOAD_MODEL).replace(".h5", "")
    
    # Diretório base para fine-tuning
    base_path = os.path.join("result", "fine_tuning", model_name)
    
    # Subpasta com data e hora
    timestamp_folder = datetime.now().strftime("%d_%m_%Y_%H%M%S")
    full_path = os.path.join(base_path, timestamp_folder)
    
    # Cria as pastas necessárias
    os.makedirs(full_path, exist_ok=True)
    os.makedirs(os.path.join(full_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(full_path, "csv"), exist_ok=True)
    os.makedirs(os.path.join(full_path, "graficos"), exist_ok=True)
    
    return full_path

def fine_tune_model(model_path, window_size, epochs, batch_size, patience, learning_rate):
    print(f"Carregando modelo existente de: {model_path}")
    
    # Carrega o modelo existente
    model = load_model(model_path)

    # Configura o otimizador com nova taxa de aprendizado
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=Settings.LOSS_FUNCTION,
        metrics=Settings.METRICS
    )

    # Carrega os dados do banco de dados
    data_df = HourlyQuote.get_between_dates(Settings.START_DATE, Settings.END_DATE)
    if data_df.empty:
        raise ValueError("Nenhum dado foi recuperado do banco de dados. Verifique a data ou os dados disponíveis.")

    # Processa indicadores técnicos, se aplicável
    data_df = TechnicalIndicators.process_indicators(data_df, Settings.INDICATORS_APPLY)
    original_timestamps = data_df['timestamp'].values
    data_df = data_df[Settings.RELEVANT_COLUMNS]

    # Prepara os dados
    processor = DataProcessor(window_size=window_size)
    X, y = processor.create_windows(
        data=data_df,
        coluna_alvo=Settings.TARGET_COLUMN,
        steps_ahead=Settings.STEPS_AHEAD
    )

    X_train, X_validation, X_test, y_train, y_validation, y_test = processor.split_data(
        X, y, train_size=Settings.TRAIN_SIZE, validation_size=Settings.VALIDATION_SPLIT
    )

    # Normaliza os dados
    X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
    X_validation_scaled, y_validation_scaled = processor.apply_normalization(X_validation, y_validation)
    X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)

    processor.save_scaler()

    # Configura callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)

    # Fine-tuning
    print("Iniciando fine-tuning...")
    history = model.fit(
        X_train_scaled, y_train_scaled,
        validation_data=(X_validation_scaled, y_validation_scaled),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    # Salva resultados
    base_path = create_finetuning_folder()
    model_save_path = os.path.join(base_path, "models", "fine_tuned_model.h5")
    model.save(model_save_path)

    # Previsão
    y_pred_scaled = model.predict(X_test_scaled)
    processor.load_scaler()
    y_pred = processor.inverse_transform(y_pred_scaled).flatten()
    y_test = processor.inverse_transform(y_test_scaled).flatten()

    # Salva previsões em CSV
    csv_exporter = CSVExporter()
    results_df = csv_exporter.save_predictions_to_csv(
        timestamps=original_timestamps[-len(y_test):],
        y_test=y_test,
        y_pred=y_pred,
        output_path=os.path.join(base_path, "csv", "fine_tuning_results.csv")
    )

    # Gera gráficos
    plotter = Plotter()
    plotter.plot_price_predictions(
        results_df=results_df,
        timestamp_col="timestamp",
        real_col="real_value",
        pred_col="predicted_value",
        save_path=os.path.join(base_path, "graficos", "price_predictions.png")
    )
    plotter.plot_errors_over_time(
        y_test=y_test,
        y_pred=y_pred,
        save_path=os.path.join(base_path, "graficos", "errors_over_time.png")
    )
    plotter.plot_histogram_of_errors(
        y_test=y_test,
        y_pred=y_pred,
        save_path=os.path.join(base_path, "graficos", "histogram_errors.png")
    )
    plotter.plot_scatter_real_vs_predicted(
        y_test=y_test,
        y_pred=y_pred,
        save_path=os.path.join(base_path, "graficos", "scatter_real_vs_predicted.png")
    )

    print(f"Fine-tuning concluído. Modelo salvo em: {model_save_path}")

if __name__ == "__main__":
    args = parse_args()
    try:
        fine_tune_model(
            model_path=args.model_path,
            window_size=args.window_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            learning_rate=args.learning_rate
        )
    except Exception as e:
        print(f"Erro durante o fine-tuning: {e}")
