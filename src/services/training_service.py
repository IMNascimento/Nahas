import os
import json
from datetime import datetime
import importlib
from database.model_nahas import TrainingRun, Experiment
from utils.csv_exporter import CSVExporter
from utils.plotter import Plotter
from data.data_processing import DataProcessor
from config.settings import Settings

def get_trainer_class(framework: str, model_type: str):
    """
    Importa e retorna a classe trainer correta conforme o framework e model_type.
    Exemplo: framework='keras', model_type='lstm'
    → models.keras.lstm_trainer.KerasLSTMTrainer
    """
    module_path = f"models.{framework.lower()}.{model_type.lower()}_trainer"
    class_name = f"{framework.capitalize()}{model_type.capitalize()}Trainer"
    try:
        module = importlib.import_module(module_path)
        trainer_class = getattr(module, class_name)
        return trainer_class
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Trainer não encontrado: {module_path}.{class_name}\nErro: {e}")

def train_model_from_config(
    config: dict,
    framework: str = "keras",
    model_type: str = "lstm",
    save_results_db: bool = True,
    save_results_csv: bool = True,
    user: str = None
):
    """
    Treina qualquer modelo suportado (Keras, PyTorch, TensorFlow) via config.
    Salva métricas, plots, CSV, banco.
    """
    # 1. Hiperparâmetros básicos
    window_size = config.get("window_size", 48)
    unique_hash = config.get("hash_config") or datetime.now().strftime("%Y%m%d%H%M%S")
    base_path = os.path.join("result", "train", unique_hash)
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "csv"), exist_ok=True)
    os.makedirs(os.path.join(base_path, "graficos"), exist_ok=True)

    model_path = os.path.join(base_path, "models", f"model_{unique_hash}.h5")
    csv_output_path = os.path.join(base_path, "csv", f"results_{unique_hash}.csv")
    plot_output_path = os.path.join(base_path, "graficos/")

    # 2. Dados
    from database.model_binance import HourlyQuoteBitcoin
    from utils.technical_indicators import TechnicalIndicators
    data_df = HourlyQuoteBitcoin.get_between_dates(Settings.START_DATE, Settings.END_DATE)
    data_df = TechnicalIndicators.process_indicators(data_df, Settings.INDICATORS_APPLY)
    original_timestamps = data_df['timestamp'].values
    data_df = data_df[Settings.RELEVANT_COLUMNS]

    processor = DataProcessor(window_size=window_size)
    X, y = processor.create_windows(
        data=data_df,
        coluna_alvo=Settings.TARGET_COLUMN,
        steps_ahead=Settings.STEPS_AHEAD
    )
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
        X, y, train_size=Settings.TRAIN_SIZE, validation_size=Settings.VALIDATION_SPLIT
    )
    X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
    X_val_scaled, y_val_scaled = processor.apply_normalization(X_val, y_val)
    X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)
    processor.save_scaler()

    # 3. Treinador dinâmico
    trainer_class = get_trainer_class(framework, model_type)
    # **Atenção:** Ajuste os nomes dos argumentos conforme suas classes!
    trainer = trainer_class(
        input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),
        **config
    )

    # 4. Treinar
    model = trainer.train(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
    trainer.save_model(model, model_path)
    # Pegue as métricas (ajuste conforme framework)
    train_loss = getattr(model, "history", {}).get("loss", [None])[-1] if hasattr(model, "history") else None
    val_loss = getattr(model, "history", {}).get("val_loss", [None])[-1] if hasattr(model, "history") else None
    # Se for PyTorch, pegue do retorno da função

    # 5. Previsão/avaliação
    if hasattr(model, "predict"):
        y_pred_scaled = model.predict(X_test_scaled)
    elif hasattr(trainer, "predict"):
        y_pred_scaled = trainer.predict(model, X_test_scaled)
    else:
        raise RuntimeError("Seu trainer/modelo precisa de método predict")

    processor.load_scaler()
    y_pred = processor.inverse_transform(y_pred_scaled).flatten()
    y_test = processor.inverse_transform(y_test_scaled).flatten()
    timestamps = original_timestamps[-len(y_test):]

    # 6. CSV de previsões
    csv_exporter = CSVExporter()
    results_df = csv_exporter.save_predictions_to_csv(
        timestamps=timestamps,
        y_test=y_test,
        y_pred=y_pred,
        output_path=csv_output_path
    )

    # 7. Plots (ajuste os métodos conforme seu plotter)
    plotter = Plotter()
    plotter.plot_price_predictions(
        results_df=results_df,
        timestamp_col="timestamp",
        real_col="real_value",
        pred_col="predicted_value",
        save_path=f"{plot_output_path}price_predictions.png"
    )
    plotter.plot_errors_over_time(
        y_test=y_test,
        y_pred=y_pred,
        save_path=f"{plot_output_path}errors_over_time.png"
    )

    # 8. Banco de dados
    experiment = None
    run = None
    if save_results_db:
        experiment = Experiment.create(
            type="single",
            framework=framework,
            model_type=model_type,
            hyperparams_json=json.dumps(config),
            hash_config=unique_hash,
            user=user,
            status="finished",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        run = TrainingRun.create(
            experiment=experiment,
            run_uuid=unique_hash,
            start_time=datetime.now(),
            end_time=datetime.now(),
            status="finished",
            model_path=model_path,
            csv_metrics_path=csv_output_path,
            metrics_json=json.dumps({
                "train_loss": float(train_loss) if train_loss else None,
                "val_loss": float(val_loss) if val_loss else None
            }),
            best_epoch=None,
            log=None,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

    # 9. Retorno final
    return {
        "train_loss": float(train_loss) if train_loss else None,
        "val_loss": float(val_loss) if val_loss else None,
        "model_path": model_path,
        "csv_output_path": csv_output_path,
        "experiment_id": experiment.id if experiment else None,
        "training_run_id": run.id if run else None
    }
