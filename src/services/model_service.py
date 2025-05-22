import os
from datetime import datetime
import numpy as np
import pandas as pd

from services.trainer_factory import TrainerFactory
from config.settings import Settings
from data.data_processing import DataProcessor
from utils.technical_indicators import TechnicalIndicators
from utils.plotter import Plotter
from utils.csv_exporter import CSVExporter

class ModelService:
    def __init__(self):
        pass

    def _get_save_dirs(self, base_folder, hash_id=None):
        """Cria estrutura de diretórios organizada por experimento/run."""
        hash_id = hash_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join("result", base_folder, hash_id)
        os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "csv"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "graficos"), exist_ok=True)
        return base_path

    def train(
        self, config: dict, framework: str, model_type: str, X=None, y=None, run_id=None
    ):
        """
        Treina qualquer modelo, qualquer framework, usando configuração.
        - config: dicionário de hiperparâmetros.
        - framework: 'keras', 'pytorch', 'tensorflow'
        - model_type: 'lstm', 'transformer', etc.
        - X, y: opcionais (se quiser passar dados prontos).
        """
        hash_id = run_id or config.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self._get_save_dirs("train", hash_id)
        model_path = os.path.join(base_path, "models", f"model_{hash_id}")
        csv_path = os.path.join(base_path, "csv", f"results_{hash_id}.csv")

        # 1. Dados
        if X is None or y is None:
            # Carregamento padrão (você pode parametrizar isso)
            from database.model_binance import HourlyQuoteBitcoin
            data_df = HourlyQuoteBitcoin.get_between_dates(Settings.START_DATE, Settings.END_DATE)
            data_df = TechnicalIndicators.process_indicators(data_df, Settings.INDICATORS_APPLY)
            data_df = data_df[Settings.RELEVANT_COLUMNS]
            original_timestamps = data_df['timestamp'].values

            window_size = config.get("window_size") or config.get("WINDOW_SIZE") or Settings.WINDOW_SIZE
            processor = DataProcessor(window_size=window_size)
            X, y = processor.create_windows(
                data=data_df, coluna_alvo=Settings.TARGET_COLUMN, steps_ahead=Settings.STEPS_AHEAD
            )
            X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
                X, y, train_size=Settings.TRAIN_SIZE, validation_size=Settings.VALIDATION_SPLIT
            )
            X_train_scaled, y_train_scaled = processor.normalize(X_train, y_train)
            X_val_scaled, y_val_scaled = processor.apply_normalization(X_val, y_val)
            X_test_scaled, y_test_scaled = processor.apply_normalization(X_test, y_test)
        else:
            # Treinamento com dados externos (para uso futuro)
            raise NotImplementedError("Suporte a dados externos ainda não implementado.")

        # 2. Cria trainer via Factory
        TrainerClass = TrainerFactory.get_trainer(framework, model_type)
        trainer = TrainerClass(input_shape=X_train_scaled.shape[1:], **config)

        # 3. Treina modelo
        model = trainer.train(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
        trainer.save_model(model_path)
        # Métricas (Keras)
        train_loss, val_loss = None, None
        if hasattr(model, "history") and hasattr(model.history, "history"):
            train_loss = model.history.history["loss"][-1]
            val_loss = model.history.history["val_loss"][-1]
        elif hasattr(trainer, "get_last_metrics"):
            train_loss, val_loss = trainer.get_last_metrics()

        # 4. Predição/avaliação
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

        # 5. CSV + gráficos
        csv_exporter = CSVExporter()
        results_df = csv_exporter.save_predictions_to_csv(
            timestamps=timestamps, y_test=y_test, y_pred=y_pred, output_path=csv_path
        )
        plotter = Plotter()
        plotter.plot_price_predictions(
            results_df=results_df,
            timestamp_col="timestamp",
            real_col="real_value",
            pred_col="predicted_value",
            save_path=os.path.join(base_path, "graficos", "price_predictions.png")
        )
        plotter.plot_errors_over_time(
            y_test=y_test, y_pred=y_pred,
            save_path=os.path.join(base_path, "graficos", "errors_over_time.png")
        )

        # 6. Retorno
        return {
            "train_loss": float(train_loss) if train_loss else None,
            "val_loss": float(val_loss) if val_loss else None,
            "model_path": model_path,
            "csv_path": csv_path,
            "run_id": hash_id,
            "results_path": base_path,
        }

    def finetune(self, model_path, config, framework, model_type, run_id=None):
        """Universal fine-tuning para qualquer trainer."""
        # Aqui você pode copiar e adaptar lógica do método train, trocando só o carregamento do modelo e re-treinando com novos dados/epochs
        # Exemplo:
        TrainerClass = TrainerFactory.get_trainer(framework, model_type)
        trainer = TrainerClass(**config)
        model = trainer.load_model(model_path)
        # Aqui você pode fazer um trainer.fine_tune(...) se existir, ou usar train() para re-treinar
        # ...
        raise NotImplementedError("Fine-tuning universal: implemente conforme seus trainers.")

    def live_run(self, framework, model_type, model_path, symbol, interval, window_hours, relevant_columns, window_size, **kwargs):
        """
        Realiza live run (previsão online) com modelo carregado.
        """
        TrainerClass = TrainerFactory.get_trainer(framework, model_type)
        trainer = TrainerClass(**kwargs)
        model = trainer.load_model(model_path)
        from services.binance import BinanceData
        from datetime import datetime, timedelta
        import time
        binance = BinanceData()
        while True:
            agora = datetime.utcnow()
            if agora.minute != 0 or agora.second != 0 or agora.microsecond != 0:
                agora -= timedelta(hours=1)
            delta = timedelta(minutes=agora.minute, seconds=agora.second, microseconds=agora.microsecond)
            end_time = agora - delta
            start_time = end_time - timedelta(hours=window_hours)
            start_str = start_time.strftime("%d %b, %Y %H:%M:%S")
            end_str = end_time.strftime("%d %b, %Y %H:%M:%S")
            df = binance.get_historical_data(symbol, start_str=start_str, interval=interval, end_str=end_str)
            for column in ['open', 'high', 'low', 'close', 'volume']:
                df[column] = pd.to_numeric(df[column], errors='coerce')
            features = df[relevant_columns].values
            processor = DataProcessor(window_size=window_size)
            features_normalized = processor.normalize_sliding_window(features)
            window_normalized = features_normalized[-window_size:]
            X_input = np.array(window_normalized, dtype=np.float32).reshape(1, window_normalized.shape[0], window_normalized.shape[1])
            # Predição universal
            if hasattr(model, "predict"):
                predicted_scaled = model.predict(X_input)
            elif hasattr(trainer, "predict"):
                predicted_scaled = trainer.predict(model, X_input)
            else:
                raise RuntimeError("Trainer/modelo não possui método predict.")
            mean = features[-window_size:].mean(axis=0)
            std = features[-window_size:].std(axis=0) + 1e-8
            predicted_value = (predicted_scaled * std[-1]) + mean[-1]
            print(f"Previsão: {predicted_value[0][0]:.2f} | Último Close: {df.iloc[-1]['close']:.2f}")
            # Espera até próximo candle
            now = datetime.utcnow()
            next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
            sleep_time = (next_hour - now).total_seconds()
            print(f"Aguardando {round(sleep_time/60, 2)} minutos até a próxima previsão.")
            time.sleep(max(sleep_time, 60))
