import os
import sys
src_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.append(src_root) if src_root not in sys.path else None
import argparse
import tensorflow as tf
from keras.callbacks import EarlyStopping
from config.settings import *
from models.db.model_binance import HourlyQuote
from data.data_processing import DataProcessor
from models.lstm_model import CustomLSTMTrainer
from utils.plotter import Plotter
from utils.csv_exporter import CSVExporter
from utils.technical_indicators import TechnicalIndicators
import hashlib
import time
import pandas as pd

set_seed(Settings.SEED)

def parse_args():
    """
    Define os argumentos que podem ser passados via terminal.
    """
    parser = argparse.ArgumentParser(description="Treinar modelo LSTM com hiperparâmetros definidos.")
    
    # Parâmetros do modelo
    parser.add_argument("--window_size", type=int, default=Settings.WINDOW_SIZE,  help="Tamanho da janela temporal.")

    # Hiperparâmetros
    parser.add_argument("--layers_config", type=str,  help="Configuração das camadas LSTM. Ex.: '[256, 128]'")
    parser.add_argument("--dropout", type=float, default=Settings.DROPOUT,  help="Taxa de dropout.")
    parser.add_argument("--recurrent_dropout", type=float,default=Settings.RECURRENT_DROPOUT, help="Taxa de dropout recorrente.")
    parser.add_argument("--batch_size", type=int, default=Settings.BATCH_SIZE, help="Tamanho do batch.")
    parser.add_argument("--epochs", type=int, default=Settings.EPOCHS,  help="Número de épocas de treinamento.")
    parser.add_argument("--patience", type=int, default=Settings.PATIENCE, help="Número de épocas para early stopping.")
    parser.add_argument("--loss", type=str, default=Settings.LOSS_FUNCTION,  help="Função de perda (ex.: 'mean_squared_error').")
    parser.add_argument("--metrics", type=str, nargs="+", default=Settings.METRICS, help="Lista de métricas para monitorar.")
    parser.add_argument("--optimizer", type=str, default=Settings.OPTIMIZER, help="Otimizador a ser usado (ex.: 'adam').")
    parser.add_argument("--l1_reg", type=float, default=Settings.L1_REGULARIZATION, help="Regularização L1.")
    parser.add_argument("--l2_reg", type=float, default=Settings.L2_REGULARIZATION, help="Regularização L2.")
    parser.add_argument("--bidirectional", default=Settings.BIDIRECTIONAL, help="Se definido, usa camadas bidirecionais.")
    parser.add_argument("--activation_functions", type=str, nargs="+", default=Settings.ACTIVATION_FUNCTION,  help="Funções de ativação para as camadas. Ex.: 'tanh relu'")
    parser.add_argument("--learning", type=float, default=Settings.LEARNING_RATE, help="Taxa de aprendizado.")
    parser.add_argument("--output_units", type=int, default=Settings.OUTPUT_UNITS, help="Número de unidades de saída do modelo.")
    return parser.parse_args()


def create_train_folder(config_hash):
    base_path = os.path.join("result/train", config_hash)
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "csv"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "graficos"), exist_ok=True)
    return base_path

def save_training_info(
        global_csv_path,
        window_size, 
        layers_config,
        dropout,
        recurrent_dropout,
        batch_size,
        epochs,
        patience,
        loss_fn,
        metrics,
        optimizer,
        l1_reg,
        l2_reg,
        bidirectional,
        activation_functions,
        learning,
        output_units,
        train_loss,
        val_loss,
        model_path):
    """
    Salva as informações do treinamento no arquivo CSV global.
    """
    file_exists = os.path.exists(global_csv_path)
    results = {
        "Bidirecional":bidirectional,
        "L1_REG": l1_reg,
        "L2_REG": l2_reg,
        "Learning_RATE":learning,
        "OUTPUT_UNITS": output_units,
        "epochs": epochs,
        "patience":patience,
        "window_size": window_size,
        "layers_config": layers_config,
        "activation_function":activation_functions,
        "Recorrente_Dropout":recurrent_dropout,
        "dropout": dropout,
        "batch_size": batch_size,
        "optimizer": optimizer,
        "loss_function": loss_fn,
        "metrics": metrics,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "model_path": model_path,
        "SEED": Settings.SEED,
        "COLUNAS": Settings.RELEVANT_COLUMNS,
        "ALVO": Settings.TARGET_COLUMN,
        "VALIDATION_SPLIT": Settings.VALIDATION_SPLIT,
        "TRAIN_SIZE": Settings.TRAIN_SIZE,
        "STEPS_AHEAD": Settings.STEPS_AHEAD,
        "GPU": Settings.USE_GPU,
        "END_DATE": Settings.END_DATE, 
        "START_DATE": Settings.START_DATE
    }
    results_df = pd.DataFrame([results])
    results_df.to_csv(global_csv_path, mode="a", header=not file_exists, index=False)

def select_device():
    """
    Define o dispositivo de treinamento com base em Settings.USE_GPU.
    """
    if Settings.USE_GPU:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("Treinando na GPU.")
            return tf.device('/GPU:0')
        else:
            print("GPU não disponível. Usando CPU.")
    else:
        print("Forçando uso da CPU.")
    
    return tf.device('/CPU:0')

def train_model(
    window_size, 
    csv_output_path,
    model_path,
    plot_output_path,
    layers_config,
    dropout,
    recurrent_dropout,
    batch_size,
    epochs,
    patience,
    loss_fn,
    metrics,
    optimizer,
    l1_reg,
    l2_reg,
    bidirectional,
    activation_functions,
    learning,
    output_units
):
    print(f"Treinando modelo com DROPOUT={dropout}, BATCH_SIZE={batch_size}, EPOCHS={epochs}, LAYERS={layers_config}")
    global_csv_path = os.path.join("result/train", "training_results.csv")
    data_df = HourlyQuote.get_between_dates(Settings.START_DATE,Settings.END_DATE)
    if data_df.empty:
        raise ValueError("Nenhum dado foi recuperado do banco de dados. Verifique a data ou os dados disponíveis.")
    
    data_df = TechnicalIndicators.process_indicators(data_df, Settings.INDICATORS_APPLY)
    original_timestamps = data_df['timestamp'].values
    data_df = data_df[Settings.RELEVANT_COLUMNS]
    processor = DataProcessor(window_size=window_size)
    X, y = processor.create_windows(
        data=data_df,
        coluna_alvo=Settings.TARGET_COLUMN,
        steps_ahead=Settings.STEPS_AHEAD
    )
    X_train, X_validation, X_test, y_train, y_validation, y_test = processor.split_data(X, y, train_size=Settings.TRAIN_SIZE ,validation_size=Settings.VALIDATION_SPLIT)
    # Normalização
    X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
    X_validation_scaled, y_validation_scaled = processor.apply_normalization(X_validation, y_validation)
    X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)
    processor.save_scaler()
    # Configuração do modelo
    input_shape = (X_train_scaled.shape[1], X_train_scaled.shape[2])
    trainer = CustomLSTMTrainer(
        input_shape=input_shape,
        layers_config=layers_config,
        dropout=dropout,
        batch_size=batch_size,
        epochs=epochs,
        patience=patience,
        recurrent_dropout=recurrent_dropout,
        loss_fn=loss_fn,
        metrics=metrics,
        optimizer=optimizer,
        l1_reg=l1_reg,
        l2_reg=l2_reg,
        bidirectional=bidirectional,
        activation_functions=activation_functions,
        output_units=output_units,
        learning_rate=learning
    )
    # Converte tudo para float32 antes de treinar (caso seja NumPy)
    X_train_scaled = tf.convert_to_tensor(X_train_scaled, dtype=tf.float32)
    y_train_scaled = tf.convert_to_tensor(y_train_scaled, dtype=tf.float32)
    X_validation_scaled = tf.convert_to_tensor(X_validation_scaled, dtype=tf.float32)
    y_validation_scaled = tf.convert_to_tensor(y_validation_scaled, dtype=tf.float32)
        
    with select_device():
        # Treina o modelo
        lstm_model = trainer.train(
            X_train_scaled, y_train_scaled,
            X_validation_scaled, y_validation_scaled
        )
    
    # Salva o modelo
    trainer.save_model(lstm_model, model_path=model_path)

    # Acessa o histórico de treinamento
    history = lstm_model.history

    # Obter train_loss e val_loss da última época
    train_loss = history.history['loss'][-1]
    val_loss = history.history['val_loss'][-1]

    # Salvar informações do treinamento no csv
    save_training_info(global_csv_path,
                        window_size, 
                        layers_config,
                        dropout,
                        recurrent_dropout,
                        batch_size,
                        epochs,
                        patience,
                        loss_fn,
                        metrics,
                        optimizer,
                        l1_reg,
                        l2_reg,
                        bidirectional,
                        activation_functions,
                        learning,
                        output_units,
                        train_loss, 
                        val_loss, 
                        model_path)

    # Previsão
    X_test_scaled = tf.convert_to_tensor(X_test_scaled, dtype=tf.float32)
    y_pred_scaled = lstm_model.predict(X_test_scaled)
    # Carrega scalers e inverte normalização
    processor.load_scaler()
    y_pred = processor.inverse_transform(y_pred_scaled).flatten()
    y_test = processor.inverse_transform(y_test_scaled).flatten()
    timestamps = original_timestamps[-len(y_test):]
    assert len(timestamps) == len(y_test) == len(y_pred), (
        f"Dimensões incompatíveis: timestamps({len(timestamps)}), "
        f"y_test({len(y_test)}), y_pred({len(y_pred)})"
    )
    
    csv_exporter = CSVExporter()
    results_df = csv_exporter.save_predictions_to_csv(
        timestamps=timestamps,
        y_test=y_test,
        y_pred=y_pred,
        output_path=csv_output_path
    )

    plotter = Plotter()
    # Plot de previsões
    plotter.plot_price_predictions(
        results_df=results_df,
        timestamp_col="timestamp",
        real_col="real_value",
        pred_col="predicted_value",
        save_path=f"{plot_output_path}price_predictions.png"
    )

    # Gera matriz de correlação
    plotter.plot_correlation_matrix(
        df=data_df,
        columns=Settings.RELEVANT_COLUMNS,
        save_path=f"{plot_output_path}correlation_matrix.png"
    )

    # Gera gráficos de erro
    plotter.plot_errors_over_time(
        y_test=y_test,
        y_pred=y_pred,
        save_path=f"{plot_output_path}errors_over_time.png"
    )

    plotter.plot_histogram_of_errors(
        y_test=y_test,
        y_pred=y_pred,
        save_path=f"{plot_output_path}histogram_errors.png"
    )

    plotter.plot_scatter_real_vs_predicted(
        y_test=y_test,
        y_pred=y_pred,
        save_path=f"{plot_output_path}scatter_real_vs_predicted.png"
    )

    print(f"Modelo salvo em: {model_path}")
    print(f"Resultados salvos em: {csv_output_path}")


if __name__ == "__main__":
    # Carrega os parâmetros com prioridade para CLI, fallback para Settings
    args = parse_args()
    layers_config = eval(args.layers_config) if args.layers_config else Settings.LAYERS_CONFIG


    # Adiciona o timestamp atual aos argumentos
    unique_input = str(args) + str(time.time())
    config_hash = hashlib.md5(unique_input.encode()).hexdigest()[:8]
    train_folder = create_train_folder(config_hash)

    # Caminhos
    model_path = os.path.join(train_folder, "models", f"model_{config_hash}.h5")
    csv_output_path = os.path.join(train_folder, "csv", f"results_{config_hash}.csv")
    plot_output_path = os.path.join(train_folder, "graficos/")

    try:
        train_model(
            window_size=args.window_size,
            csv_output_path=csv_output_path,
            model_path=model_path,
            plot_output_path=plot_output_path,
            layers_config=layers_config,
            dropout= args.dropout,
            recurrent_dropout=args.recurrent_dropout,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            loss_fn=args.loss,
            metrics=args.metrics,
            optimizer=args.optimizer,
            l1_reg=args.l1_reg,
            l2_reg=args.l2_reg,
            bidirectional=args.bidirectional,
            activation_functions=args.activation_functions,
            output_units=args.output_units,
            learning=args.learning
        )
    except Exception as e:
        print(f"Erro ao treinar o modelo: {e}")

