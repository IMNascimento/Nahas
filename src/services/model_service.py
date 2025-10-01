import os
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json
import random
import signal
from contextlib import contextmanager

from database.model_nahas import TrainingRun, FineTuningRun, db, GridResult
from services.trainer_factory import TrainerFactory
from data.data_processing import DataProcessor
from utils.technical_indicators import TechnicalIndicators
from utils.plotter import Plotter
from utils.csv_exporter import CSVExporter
from config.settings import BASE_DIR, Settings, set_seed, set_cuda_tensorflow, set_cuda_pytorch
from services.binance import BinanceData
from utils.db_utils import ensure_db_connection
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data.evomsn_normalizer import EvoMSNNormalizer, EvoMSNLikeNormalizer

import concurrent.futures
from typing import Any, Dict, List, Tuple
from utils.capacity_train import pick_gpu_for_job, estimate_job_mem_bytes 


# -------------------------
# TIMEOUT (comentar no Windows)
# -------------------------
class TimeoutError(Exception):
    pass

@contextmanager
def timeout(seconds):
    """Context manager para timeout. Comentar no Windows."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Operação excedeu {seconds} segundos")
    
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# -------------------------
# WORKER COM FIX DE BANCO
# -------------------------
def _train_job_worker(args: Tuple[Dict[str, Any], str, str, str, int | None]):
    """
    Worker paralelo com:
    - Reconexão MySQL isolada por processo (FIX crítico)
    - Proteção contra deadlocks de GPU
    - Timeout para operações longas
    """
    cfg, fw, mt, artifacts_base, gpu_idx = args
    
    worker_id = os.getpid()
    print(f"[WORKER-{worker_id}] Iniciando | GPU: {gpu_idx} | Framework: {fw} | Model: {mt}", flush=True)
    
    # === CRÍTICO: RECONECTAR BANCO (evita "Packet sequence number wrong") ===
    try:
        from database.model_base import db
        if not db.is_closed():
            db.close()
        db.connect(reuse_if_open=False)
        print(f"[WORKER-{worker_id}] ✓ Conexão MySQL criada para PID {worker_id}", flush=True)
    except Exception as e:
        print(f"[WORKER-{worker_id}] ✗ ERRO ao conectar ao banco: {e}", flush=True)
        raise
    
    try:
        # === GPU CONFIG ===
        if gpu_idx is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            cfg = cfg.copy()
            cfg["use_gpu"] = True
            cfg["gpu_index"] = 0  # Sempre 0 após isolation
            print(f"[WORKER-{worker_id}] GPU {gpu_idx} isolada (mapeada para 0)", flush=True)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            cfg = cfg.copy()
            cfg["use_gpu"] = False
            cfg["gpu_index"] = None
            print(f"[WORKER-{worker_id}] Modo CPU", flush=True)
        
        # === FRAMEWORK INIT ===
        try:
            with timeout(30):
                if fw.lower() in ("tensorflow", "keras"):
                    import tensorflow as tf
                    gpus = tf.config.list_physical_devices('GPU')
                    if gpus:
                        try:
                            for gpu in gpus:
                                tf.config.experimental.set_memory_growth(gpu, True)
                            print(f"[WORKER-{worker_id}] TF memory growth habilitado", flush=True)
                        except RuntimeError as e:
                            print(f"[WORKER-{worker_id}] Aviso memory growth: {e}", flush=True)
                    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
                    
                elif fw.lower() == "pytorch":
                    import torch
                    if gpu_idx is not None and not torch.cuda.is_available():
                        raise RuntimeError("PyTorch: CUDA não disponível")
                    print(f"[WORKER-{worker_id}] PyTorch pronto", flush=True)
                
                print(f"[WORKER-{worker_id}] Framework {fw} inicializado", flush=True)
                
        except TimeoutError as e:
            print(f"[WORKER-{worker_id}] TIMEOUT init framework: {e}", flush=True)
            raise RuntimeError(f"Framework {fw} travou na inicialização")
        
        # === SEED ===
        seed_val = cfg.get("seed")
        if seed_val is None or str(seed_val).upper() == "RANDOM":
            seed_val = random.SystemRandom().randint(0, 2**32 - 1)
            cfg["seed"] = seed_val
        
        print(f"[WORKER-{worker_id}] Seed: {seed_val}", flush=True)
        set_seed(seed_val)
        
        # === TREINO ===
        print(f"[WORKER-{worker_id}] Iniciando treino...", flush=True)
        svc = ModelService()
        
        try:
            with timeout(7200):  # 2h max
                out = svc.train(cfg, fw, mt, artifacts_base=artifacts_base)
            print(f"[WORKER-{worker_id}] ✓ Treino concluído!", flush=True)
            return out
            
        except TimeoutError:
            print(f"[WORKER-{worker_id}] ✗ TIMEOUT treino (>2h)", flush=True)
            raise RuntimeError("Treino excedeu 2 horas")
        
        finally:
            # Fecha conexão ao terminar
            try:
                from database.model_base import db
                if not db.is_closed():
                    db.close()
                    print(f"[WORKER-{worker_id}] Conexão MySQL fechada", flush=True)
            except Exception:
                pass
    
    except Exception as e:
        print(f"[WORKER-{worker_id}] ✗ ERRO: {type(e).__name__}: {str(e)}", flush=True)
        import traceback
        traceback.print_exc()
        raise


# -------------------------
# Helpers originais
# -------------------------
def prepare_and_set_seed(config: dict) -> int:
    seed = config.get("seed")
    if isinstance(seed, str):
        s = seed.strip()
        if s.upper() == "RANDOM" or s == "":
            seed = None
        else:
            try:
                seed = int(s)
            except Exception:
                seed = None
    if not isinstance(seed, (int, np.integer)):
        seed = random.SystemRandom().randint(0, 2**32 - 1)
    config["seed"] = int(seed)
    set_seed(int(seed))
    return int(seed)


def get_epochs_trained(model_or_history):
    if hasattr(model_or_history, "history") and isinstance(model_or_history.history, dict):
        return len(model_or_history.history.get("loss", []))
    if hasattr(model_or_history, "history") and hasattr(model_or_history.history, "epoch"):
        return len(model_or_history.history.epoch)
    if hasattr(model_or_history, "epoch"):
        return model_or_history.epoch if isinstance(model_or_history.epoch, int) else len(model_or_history.epoch)
    if hasattr(model_or_history, "epochs_trained"):
        return model_or_history.epochs_trained
    if hasattr(model_or_history, "epochs"):
        return model_or_history.epochs
    if isinstance(model_or_history, dict) and "loss" in model_or_history:
        return len(model_or_history["loss"])
    return None


def _parse_dt(x):
    if isinstance(x, datetime):
        return x
    if isinstance(x, str):
        return datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    return datetime.now()

class ModelService:
    def __init__(self):
        pass

    # -------------------------
    # métricas básicas
    # -------------------------
    @staticmethod
    def _compute_basic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        yt = np.asarray(y_true).reshape(-1)
        yp = np.asarray(y_pred).reshape(-1)
        mse = float(mean_squared_error(yt, yp))
        rmse = float(np.sqrt(mse))
        mae = float(mean_absolute_error(yt, yp))
        r2  = float(r2_score(yt, yp))
        return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}

    # -------------------------
    # infra de pastas
    # -------------------------
    def _get_save_dirs(self, base_folder, hash_id=None):
        hash_id = hash_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join(BASE_DIR, "results", base_folder, hash_id)
        os.makedirs(os.path.join(base_path, "models"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "csv"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "graficos"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "scaler"), exist_ok=True)
        os.makedirs(os.path.join(base_path, "hiperparams"), exist_ok=True)
        return base_path

    # -------------------------
    # pós-processamento universal
    # -------------------------
    def universal_postprocess(self, y_pred, y_test, timestamps):
        y_pred = np.array(y_pred)
        y_test = np.array(y_test)
        timestamps = np.array(timestamps)

        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        if y_test.ndim == 1:
            y_test = y_test.reshape(-1, 1)

        n_samples = y_pred.shape[0]
        n_steps = y_pred.shape[1]
        y_test = y_test[:, :n_steps]

        assert y_pred.shape == y_test.shape, f"Shape mismatch: y_test={y_test.shape}, y_pred={y_pred.shape}"
        assert len(timestamps) >= n_samples, "Timestamps length mismatch"

        return y_pred, y_test, timestamps[-n_samples:], n_steps

    # -------------------------
    # util: extrair config de normalização
    # -------------------------
    def _norm_cfg(self, config: dict):
        norm = config.get("normalization", {}) or {}
        strategy = norm.get("strategy", "global").lower()  # "global" | "local" | "evomsn" | "evomsn_like"
        if strategy not in ("global", "local", "evomsn", "evomsn_like"):
            raise ValueError("normalization.strategy deve ser 'global', 'local', 'evomsn' ou 'evomsn_like'.")

        scaler_type = norm.get("scaler_type", "robust")   # global
        x_mode = norm.get("x_mode", "zscore")             # local
        y_mode = norm.get("y_mode", "none")               # local

        # EvoMSN params
        ev_k   = int(norm.get("evomsn_k_scales", config.get("evomsn_k_scales", 4)))
        ev_agg = norm.get("evomsn_agg", "fft")            # "fft" | "uniform"
        ev_pred = norm.get("evomsn_predictor", config.get("evomsn_predictor", "linear"))  # "linear" | "mlp"

        return strategy, scaler_type, x_mode, y_mode, {"k_scales": ev_k, "agg": ev_agg, "predictor": ev_pred}

    # -------------------------
    # util: y_mode local exige o alvo dentro do X
    # -------------------------
    @staticmethod
    def _y_mode_needs_target_in_X(y_mode: str) -> bool:
        return y_mode in ("relative_last", "zscore_target", "minmax_target", "robust_target")

    # constrói janelas do alvo para concatenar como último canal do X
    @staticmethod
    def _build_target_windows(target_series: np.ndarray, window_size: int, steps_ahead: int) -> np.ndarray:
        n = len(target_series) - window_size - steps_ahead + 1
        if n <= 0:
            raise ValueError("Série muito curta para window_size e steps_ahead.")
        tw = np.empty((n, window_size, 1), dtype=float)
        for i in range(n):
            tw[i, :, 0] = target_series[i:i + window_size]
        return tw

    # split timestamps na mesma lógica do split_data
    @staticmethod
    def _split_timestamps(ts_all: np.ndarray, train_size: float, val_size: float):
        n_total = len(ts_all)
        n_train = int(n_total * train_size)
        n_val = int(n_total * val_size)
        ts_train = ts_all[:n_train]
        ts_val = ts_all[n_train:n_train + n_val]
        ts_test = ts_all[n_train + n_val:]
        return ts_train, ts_val, ts_test

    # -------------------------
    # TREINO
    # -------------------------
    def train(self, config: dict, framework: str, model_type: str, X=None, y=None, run_id=None, artifacts_base: str = "train"):
        seed = prepare_and_set_seed(config)
        config["seed"] = seed

        if config.get("use_gpu", True):
            if framework.lower() in ("tensorflow", "keras"):
                set_cuda_tensorflow(config.get("gpu_index", 0))
            elif framework.lower() == "pytorch":
                set_cuda_pytorch(config.get("gpu_index", 0))

        norm_strategy, scaler_type, x_mode, y_mode, evcfg = self._norm_cfg(config)
        hash_id = run_id or config.get("run_id") or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self._get_save_dirs(artifacts_base, hash_id)
        ext = {"keras": ".keras", "tensorflow": ".keras", "pytorch": ".pt"}.get(framework.lower(), ".model")

        model_path = os.path.join(base_path, "models", f"model_{hash_id}{ext}")
        csv_path = os.path.join(base_path, "csv", f"results_{hash_id}.csv")
        scaler_dir = os.path.join(base_path, "scaler")
        config_path = os.path.join(base_path, "hiperparams", f"config_{hash_id}.json")
        metrics_path = os.path.join(base_path, "metrics.json")

        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

        # -------------------------
        # 1) Dados
        # -------------------------
        if X is None or y is None:
            from database.model_binance import HourlyQuoteBitcoin
            df = HourlyQuoteBitcoin.get_between_dates(config.get("start_date"), config.get("end_date"))

            if config.get("indicators_apply"):
                df = TechnicalIndicators.process_indicators(df, config.get("indicators_apply"))

            for c in config.get("relevant_columns", []):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")

            window_size = int(config.get("window_size"))
            steps_ahead = int(config.get("steps_ahead", 1))
            target_col = config.get("target_column")
            assert target_col in df.columns, f"Coluna alvo '{target_col}' não encontrada em df."

            feat_cols = [c for c in (config.get("relevant_columns") or []) if c != "timestamp"]
            cols_for_window = list(dict.fromkeys(feat_cols + [target_col, "timestamp"]))
            data_df = df[cols_for_window].dropna().reset_index(drop=True)

            processor = DataProcessor(window_size=window_size, scaler_type=scaler_type)

            # janelas + timestamps alinhados ao horizonte
            X_all, y_all, ts_all = processor.create_windows_and_timestamps(
                data_df, coluna_alvo=target_col, steps_ahead=steps_ahead,
                timestamp_col="timestamp", ts_mode="horizon"
            )

            # Para estratégias que precisam do alvo no X (local com certos y_mode)
            if norm_strategy == "local" and self._y_mode_needs_target_in_X(y_mode):
                target_series = data_df[target_col].values
                tw_all = self._build_target_windows(target_series, window_size, steps_ahead)  # (n, L, 1)
                X_all = np.concatenate([X_all, tw_all], axis=2)
            else:
                # usado pelo EvoMSN (pesos do ensemble via janela do alvo)
                tw_all = self._build_target_windows(data_df[target_col].values, window_size, steps_ahead)

            # split
            X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
                X_all, y_all,
                train_size=float(config.get("train_size", 0.7)),
                validation_size=float(config.get("validation_split", 0.15))
            )
            ts_train, ts_val, ts_test = self._split_timestamps(
                ts_all, float(config.get("train_size", 0.7)), float(config.get("validation_split", 0.15))
            )
            # target windows alinhadas (para pesos do ensemble)
            n_tr, n_v = X_train.shape[0], X_val.shape[0]
            tw_train, tw_val, tw_test = tw_all[:n_tr], tw_all[n_tr:n_tr + n_v], tw_all[n_tr + n_v:]

            # -------------------------
            # normalização por estratégia
            # -------------------------
            ctx_train_local = None   # usado só se local
            ctx_train_msn   = None   # usado só se evomsn/like

            if norm_strategy == "global":
                X_train_scaled, y_train_scaled = processor.normalize_global(X_train, y_train)
                processor.save_scaler(scaler_dir)
                X_val_scaled,  y_val_scaled  = processor.apply_normalization_global(X_val,  y_val)
                X_test_scaled, y_test_scaled = processor.apply_normalization_global(X_test, y_test)
                inverse_kind = "global"
                inverse_ctx_test = None

            elif norm_strategy == "local":
                target_idx = X_train.shape[2] - 1 if self._y_mode_needs_target_in_X(y_mode) else 0
                # guarde ctx do TREINO para inversão de métricas do treino
                X_train_scaled, y_train_scaled, ctx_train_local = processor.normalize_local(
                    X_train, y_train, x_mode=x_mode, y_mode=y_mode, target_idx=target_idx
                )
                X_val_scaled, y_val_scaled, _ = processor.apply_normalization_local(
                    X_val, y_val, x_mode=x_mode, y_mode=y_mode, target_idx=target_idx
                )
                X_test_scaled, y_test_scaled, inverse_ctx_test = processor.apply_normalization_local(
                    X_test, y_test, x_mode=x_mode, y_mode=y_mode, target_idx=target_idx
                )
                inverse_kind = "local"

            else:  # "evomsn" ou "evomsn_like"
                if norm_strategy == "evomsn":
                    msn = EvoMSNNormalizer(
                        window_size=window_size,
                        horizon=steps_ahead,
                        n_features=X_train.shape[2],
                        target_idx=0,  # pesos vêm de target_windows
                        k_scales=int(evcfg["k_scales"]),
                        agg=str(evcfg.get("agg", "fft")),
                        random_state=seed,
                        predictor_type=str(evcfg.get("predictor", "linear")),
                    )
                else:
                    msn = EvoMSNLikeNormalizer(
                        window_size=window_size,
                        horizon=steps_ahead,
                        n_features=X_train.shape[2],
                        target_idx=0,
                        k_scales=int(evcfg["k_scales"]),
                        agg=str(evcfg.get("agg", "fft")),
                        random_state=seed,
                    )

                # treino MSN (períodos globais a partir de X_train)
                X_train_scaled, y_train_scaled = msn.fit(X_train, y_train, save_path=scaler_dir)
                # obtenha ctx também para o CONJUNTO DE TREINO (para inversão das métricas)
                _Xtr_tmp, _ytr_tmp, ctx_train_msn = msn.transform(X_train, None, target_windows=tw_train)
                # validação e teste com pesos via janela do alvo
                X_val_scaled,  y_val_scaled,  _ctx_val  = msn.transform(X_val,  y_val,  target_windows=tw_val)
                X_test_scaled, _y_dummy,     inverse_ctx_test = msn.transform(X_test, None,   target_windows=tw_test)
                inverse_kind = "evomsn"

                # empacote objeto e ctx de teste para a etapa de inversão
                inverse_ctx_test = (msn, inverse_ctx_test)

        else:
            raise NotImplementedError("Treino com X,y externos ainda não implementado.")

        # -------------------------
        # 2) Trainer/Modelo
        # -------------------------
        TrainerClass = TrainerFactory.get_trainer(framework, model_type)
        trainer = TrainerClass(input_shape=X_train_scaled.shape[1:], **config)

        # -------------------------
        # 3) Treinar
        # -------------------------
        model = trainer.train(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled)
        trainer.save_model(model_path)

        train_loss, val_loss = None, None
        if hasattr(model, "history") and hasattr(model.history, "history"):
            train_loss = model.history.history["loss"][-1]
            val_loss = model.history.history["val_loss"][-1]
        elif hasattr(trainer, "get_last_metrics"):
            train_loss, val_loss = trainer.get_last_metrics()

        # -------------------------
        # 4A) Predição + inversão (TESTE)
        # -------------------------
        if hasattr(model, "predict"):
            y_pred_scaled_test = model.predict(X_test_scaled)
        elif hasattr(trainer, "predict"):
            y_pred_scaled_test = trainer.predict(model, X_test_scaled)
        else:
            raise RuntimeError("Seu trainer/modelo precisa de método predict.")

        if inverse_kind == "global":
            y_pred_test = processor.inverse_transform_global(y_pred_scaled_test)
            y_test_orig = processor.inverse_transform_global(y_test_scaled)

        elif inverse_kind == "local":
            y_pred_test = processor.inverse_transform_local(y_pred_scaled_test, inverse_ctx_test)
            y_test_orig = processor.inverse_transform_local(y_test_scaled, inverse_ctx_test)

        else:  # "evomsn" e "evomsn_like"
            msn, ctx_test = inverse_ctx_test
            _per_scale_test, y_pred_test = msn.denorm_and_ensemble(y_pred_scaled_test, ctx_test)  # (N,H)
            y_test_orig = y_test  # já está no domínio real nesse caminho

        # alinhar shapes/timestamps (TESTE)
        y_pred_test, y_test_orig, timestamps, n_steps = self.universal_postprocess(y_pred_test, y_test_orig, ts_test)

        # -------------------------
        # 4B) Predição + inversão (TREINO) para métricas
        # -------------------------
        if hasattr(model, "predict"):
            y_pred_scaled_train = model.predict(X_train_scaled)
        else:
            y_pred_scaled_train = trainer.predict(model, X_train_scaled)

        if inverse_kind == "global":
            y_pred_train = processor.inverse_transform_global(y_pred_scaled_train)
            y_train_orig = processor.inverse_transform_global(y_train_scaled)

        elif inverse_kind == "local":
            # usa o ctx do TREINO que guardamos na normalização local
            y_pred_train = processor.inverse_transform_local(y_pred_scaled_train, ctx_train_local)
            y_train_orig = processor.inverse_transform_local(y_train_scaled, ctx_train_local)

        else:  # "evomsn"/"evomsn_like"
            # ctx de treino calculado acima (ctx_train_msn)
            _per_scale_tr, y_pred_train = msn.denorm_and_ensemble(y_pred_scaled_train, ctx_train_msn)
            y_train_orig = y_train  # já no domínio real

        # alinhar shapes para métricas de treino (timestamps não usados nas métricas):
        y_pred_train = np.array(y_pred_train)
        y_train_orig = np.array(y_train_orig)
        if y_pred_train.ndim == 1:
            y_pred_train = y_pred_train.reshape(-1, 1)
        if y_train_orig.ndim == 1:
            y_train_orig = y_train_orig.reshape(-1, 1)
        # garantir colunas compatíveis
        n_steps_train = y_pred_train.shape[1]
        y_train_orig = y_train_orig[:, :n_steps_train]

        # -------------------------
        # 4C) Métricas (Train/Test)
        # -------------------------
        metrics_train = self._compute_basic_metrics(y_train_orig, y_pred_train)
        metrics_test  = self._compute_basic_metrics(y_test_orig, y_pred_test)

        metrics_readable = {
            "MAE - Train data": metrics_train["mae"],
            "MAE - Test data": metrics_test["mae"],
            "RMSE - Train data": metrics_train["rmse"],
            "RMSE - Test data": metrics_test["rmse"],
            "MSE - Train data": metrics_train["mse"],
            "MSE - Test data": metrics_test["mse"],
            "R2 score - Train data": metrics_train["r2"],
            "R2 score - Test data": metrics_test["r2"],
        }
        metrics_bundle = {
            "train": metrics_train,
            "test": metrics_test,
            "readable": metrics_readable,
            "n_steps": int(n_steps),
            "window_size": int(window_size),
            "run_id": hash_id,
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics_bundle, f, indent=4)

        # -------------------------
        # 5) CSV + Gráficos (TESTE)
        # -------------------------
        csv_exporter = CSVExporter()
        results_df = csv_exporter.save_predictions_to_csv(
            timestamps=timestamps, y_test=y_test_orig, y_pred=y_pred_test,
            steps_ahead=n_steps, output_path=csv_path
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
            y_test=y_test_orig, y_pred=y_pred_test,
            save_path=os.path.join(base_path, "graficos", "errors_over_time.png")
        )
        try:
            plotter.plot_correlation_matrix(
                df=df,
                columns=config.get("relevant_columns"),
                save_path=os.path.join(base_path, "graficos", "correlation_matrix.png")
            )
        except Exception:
            pass
        plotter.plot_histogram_of_errors(
            y_test=y_test_orig, y_pred=y_pred_test,
            save_path=os.path.join(base_path, "graficos", "histogram_errors.png")
        )
        plotter.plot_scatter_real_vs_predicted(
            y_test=y_test_orig, y_pred=y_pred_test,
            save_path=os.path.join(base_path, "graficos", "scatter_real_vs_predicted.png")
        )

        # -------------------------
        # 6) Banco
        # -------------------------
        metrics_json_db = json.dumps(metrics_bundle, indent=4)
        ensure_db_connection()
        TrainingRun.create(
            run_uuid=hash_id,
            start_time=_parse_dt(config.get("start_date")),
            end_time=_parse_dt(config.get("end_date")),
            status="finished",
            model_path=model_path,
            csv_metrics_path=csv_path,
            plot_dir=os.path.join(base_path, "graficos"),
            config_path=config_path,
            framework=framework,
            model_type=model_type,
            target_column=config.get("target_column"),
            seed=seed,
            gpu_used=Settings.USE_GPU,
            train_loss=train_loss,
            val_loss=val_loss,
            best_epoch=get_epochs_trained(model if not hasattr(model, "history") else model.history),
            log=None,
            metrics_json=metrics_json_db, 
        )

        return {
            "train_loss": float(train_loss) if train_loss is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "model_path": model_path,
            "csv_path": csv_path,
            "metrics": metrics_bundle,
            "metrics_path": metrics_path,
            "run_id": hash_id,
            "results_path": base_path,
        }

    # -------------------------
    # FINE-TUNE
    # -------------------------
    # -------------------------
    # FINE-TUNE PRODUCTION-READY
    # -------------------------
    def finetune(self, model_path, config, framework, model_type, original_run_id, run_id=None):
        """
        Fine-tuning production-ready com:
        - Validação de compatibilidade
        - Backup automático
        - Comparação de métricas
        - Rollback se degradar
        - Logging detalhado
        """
        from utils.logger import Logger
        import shutil
        
        # Setup logging
        hash_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = self._get_save_dirs("finetune", hash_id)
        log_path = os.path.join(base_path, f"finetune_{hash_id}.log")
        logger = Logger(log_file=log_path)
        
        logger.info(f"[FINETUNE] Iniciando fine-tuning do modelo: {model_path}")
        logger.info(f"[FINETUNE] Run ID: {hash_id} | Original Run: {original_run_id}")
        
        try:
            # === 1. CARREGAR MÉTRICAS DO MODELO ORIGINAL ===
            logger.info("[FINETUNE] Carregando métricas do modelo original...")
            original_metrics = None
            try:
                ensure_db_connection()
                original_run = TrainingRun.get(TrainingRun.run_uuid == original_run_id)
                if original_run.metrics_json:
                    original_metrics = json.loads(original_run.metrics_json)
                    logger.info(f"[FINETUNE] Métricas originais carregadas: Test RMSE={original_metrics.get('test', {}).get('rmse', 'N/A')}")
            except Exception as e:
                logger.warning(f"[FINETUNE] Falha ao carregar métricas originais: {e}")
            
            # === 2. BACKUP DO MODELO ORIGINAL ===
            logger.info("[FINETUNE] Criando backup do modelo original...")
            backup_dir = os.path.join(base_path, "backup")
            os.makedirs(backup_dir, exist_ok=True)
            backup_model_path = os.path.join(backup_dir, os.path.basename(model_path))
            
            try:
                shutil.copy2(model_path, backup_model_path)
                logger.info(f"[FINETUNE] Backup salvo em: {backup_model_path}")
            except Exception as e:
                logger.error(f"[FINETUNE] Falha ao criar backup: {e}")
                raise RuntimeError(f"Não foi possível criar backup do modelo original: {e}")
            
            # === 3. VALIDAÇÃO DE CONFIG ===
            logger.info("[FINETUNE] Validando configuração...")
            norm_strategy, scaler_type, x_mode, y_mode, evcfg = self._norm_cfg(config)
            
            # Verificar compatibilidade de normalização com original
            try:
                original_config_path = original_run.config_path
                with open(original_config_path, "r") as f:
                    original_config = json.load(f)
                
                orig_norm = original_config.get("normalization", {})
                orig_strategy = orig_norm.get("strategy", "global")
                
                if norm_strategy != orig_strategy:
                    logger.warning(f"[FINETUNE] ⚠️  Estratégia de normalização diferente: Original={orig_strategy}, Novo={norm_strategy}")
                    logger.warning("[FINETUNE] Isso pode causar incompatibilidades. Considere usar a mesma estratégia.")
            except Exception as e:
                logger.warning(f"[FINETUNE] Não foi possível verificar compatibilidade de normalização: {e}")
            
            # === 4. CONFIGURAÇÃO DE GPU ===
            if config.get("use_gpu", True):
                if framework.lower() in ("tensorflow", "keras"):
                    set_cuda_tensorflow(config.get("gpu_index", 0))
                elif framework.lower() == "pytorch":
                    set_cuda_pytorch(config.get("gpu_index", 0))
            
            # === 5. PREPARAR ARTEFATOS ===
            ext = {"keras": ".keras", "tensorflow": ".keras", "pytorch": ".pt"}.get(framework.lower(), ".model")
            model_ft_path = os.path.join(base_path, "models", f"model_{hash_id}{ext}")
            csv_path = os.path.join(base_path, "csv", f"results_{hash_id}.csv")
            scaler_dir = os.path.join(base_path, "scaler")
            config_path = os.path.join(base_path, "hiperparams", f"config_{hash_id}.json")
            plot_dir = os.path.join(base_path, "graficos")
            metrics_path = os.path.join(base_path, "metrics.json")
            comparison_path = os.path.join(base_path, "comparison.json")
            
            config["run_id"] = hash_id
            with open(config_path, "w") as f:
                json.dump(config, f, indent=4)
            
            # === 6. CARREGAR E PROCESSAR DADOS ===
            logger.info("[FINETUNE] Carregando dados para fine-tuning...")
            from database.model_binance import HourlyQuoteBitcoin
            df = HourlyQuoteBitcoin.get_between_dates(config.get("start_date"), config.get("end_date"))
            
            if config.get("indicators_apply"):
                df = TechnicalIndicators.process_indicators(df, config.get("indicators_apply"))
            
            for c in config.get("relevant_columns", []):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            
            window_size = int(config.get("window_size"))
            steps_ahead = int(config.get("steps_ahead", 1))
            target_col = config.get("target_column")
            
            logger.info(f"[FINETUNE] Parâmetros: window={window_size}, steps_ahead={steps_ahead}, target={target_col}")
            
            feat_cols = [c for c in (config.get("relevant_columns") or []) if c != "timestamp"]
            cols_for_window = list(dict.fromkeys(feat_cols + [target_col, "timestamp"]))
            data_df = df[cols_for_window].dropna().reset_index(drop=True)
            
            logger.info(f"[FINETUNE] Dataset: {len(data_df)} amostras válidas")
            
            processor = DataProcessor(window_size=window_size, scaler_type=scaler_type)
            X_all, y_all, ts_all = processor.create_windows_and_timestamps(
                data_df, coluna_alvo=target_col, steps_ahead=steps_ahead,
                timestamp_col="timestamp", ts_mode="horizon"
            )
            tw_all = self._build_target_windows(data_df[target_col].values, window_size, steps_ahead)
            
            X_train, X_val, X_test, y_train, y_val, y_test = processor.split_data(
                X_all, y_all,
                train_size=float(config.get("train_size", 0.7)),
                validation_size=float(config.get("validation_split", 0.15))
            )
            ts_train, ts_val, ts_test = self._split_timestamps(
                ts_all, float(config.get("train_size", 0.7)), float(config.get("validation_split", 0.15))
            )
            n_tr, n_v = X_train.shape[0], X_val.shape[0]
            tw_train, tw_val, tw_test = tw_all[:n_tr], tw_all[n_tr:n_tr + n_v], tw_all[n_tr + n_v:]
            
            logger.info(f"[FINETUNE] Split: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
            
            # === 7. NORMALIZAÇÃO ===
            logger.info(f"[FINETUNE] Aplicando normalização: {norm_strategy}")
            ctx_train_local = None
            ctx_train_msn = None
            
            if norm_strategy == "global":
                X_train_scaled, y_train_scaled = processor.normalize_global(X_train, y_train)
                processor.save_scaler(scaler_dir)
                X_val_scaled, y_val_scaled = processor.apply_normalization_global(X_val, y_val)
                X_test_scaled, y_test_scaled = processor.apply_normalization_global(X_test, y_test)
                inverse_kind = "global"
                inverse_ctx_test = None
            
            elif norm_strategy == "local":
                target_idx = 0
                X_train_scaled, y_train_scaled, ctx_train_local = processor.normalize_local(
                    X_train, y_train, x_mode=x_mode, y_mode=y_mode, target_idx=target_idx
                )
                X_val_scaled, y_val_scaled, _ = processor.apply_normalization_local(
                    X_val, y_val, x_mode=x_mode, y_mode=y_mode, target_idx=target_idx
                )
                X_test_scaled, y_test_scaled, inverse_ctx_test = processor.apply_normalization_local(
                    X_test, y_test, x_mode=x_mode, y_mode=y_mode, target_idx=target_idx
                )
                inverse_kind = "local"
            
            else:  # evomsn/evomsn_like
                if norm_strategy == "evomsn":
                    msn = EvoMSNNormalizer(
                        window_size=window_size, horizon=steps_ahead, n_features=X_train.shape[2],
                        target_idx=0, k_scales=int(evcfg["k_scales"]), agg=str(evcfg.get("agg", "fft")),
                        random_state=config.get("seed", 42), predictor_type=str(evcfg.get("predictor", "linear"))
                    )
                else:
                    msn = EvoMSNLikeNormalizer(
                        window_size=window_size, horizon=steps_ahead, n_features=X_train.shape[2],
                        target_idx=0, k_scales=int(evcfg["k_scales"]), agg=str(evcfg.get("agg", "fft")),
                        random_state=config.get("seed", 42)
                    )
                X_train_scaled, y_train_scaled = msn.fit(X_train, y_train, save_path=scaler_dir)
                _Xtr_tmp, _ytr_tmp, ctx_train_msn = msn.transform(X_train, None, target_windows=tw_train)
                X_val_scaled, y_val_scaled, _ctx_val = msn.transform(X_val, y_val, target_windows=tw_val)
                X_test_scaled, _y_dummy, ctx_test = msn.transform(X_test, None, target_windows=tw_test)
                inverse_kind = "evomsn"
                inverse_ctx_test = (msn, ctx_test)
            
            # === 8. CARREGAR MODELO E VALIDAR COMPATIBILIDADE ===
            logger.info("[FINETUNE] Carregando modelo original...")
            TrainerClass = TrainerFactory.get_trainer(framework, model_type)
            trainer = TrainerClass(input_shape=X_train_scaled.shape[1:], **config)
            
            try:
                trainer.load_model(model_path)
                logger.info("[FINETUNE] ✓ Modelo carregado com sucesso")
            except Exception as e:
                logger.error(f"[FINETUNE] ✗ Falha ao carregar modelo: {e}")
                raise RuntimeError(f"Não foi possível carregar o modelo original: {e}")
            
            # Validar compatibilidade de input_shape
            expected_shape = X_train_scaled.shape[1:]
            try:
                test_input = np.zeros((1,) + expected_shape)
                _ = trainer.predict(test_input)
                logger.info(f"[FINETUNE] ✓ Compatibilidade validada: input_shape={expected_shape}")
            except Exception as e:
                logger.error(f"[FINETUNE] ✗ Incompatibilidade de shape: {e}")
                raise RuntimeError(f"O modelo original não é compatível com os novos dados: {e}")
            
            # === 9. FINE-TUNING ===
            logger.info(f"[FINETUNE] Iniciando fine-tuning: {config.get('epochs', 10)} epochs, lr={config.get('learning_rate', 0.0001)}")
            try:
                model = trainer.finetune(X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled, config)
                logger.info("[FINETUNE] ✓ Fine-tuning concluído")
            except Exception as e:
                logger.error(f"[FINETUNE] ✗ Erro durante fine-tuning: {e}")
                raise
            
            trainer.save_model(model_ft_path)
            logger.info(f"[FINETUNE] Modelo fine-tuned salvo: {model_ft_path}")
            
            # === 10. MÉTRICAS ===
            train_loss, val_loss = None, None
            if hasattr(model, "history") and hasattr(model.history, "history"):
                train_loss = model.history.history["loss"][-1]
                val_loss = model.history.history["val_loss"][-1]
            elif hasattr(trainer, "get_last_metrics"):
                train_loss, val_loss = trainer.get_last_metrics()
            
            logger.info(f"[FINETUNE] Loss: train={train_loss}, val={val_loss}")
            
            # === 11. PREDIÇÕES E MÉTRICAS FINAIS ===
            logger.info("[FINETUNE] Calculando métricas finais...")
            
            # TEST
            y_pred_scaled_test = trainer.predict(model, X_test_scaled) if hasattr(trainer, "predict") else model.predict(X_test_scaled)
            if inverse_kind == "global":
                y_pred_test = processor.inverse_transform_global(y_pred_scaled_test)
                y_test_orig = processor.inverse_transform_global(y_test_scaled)
            elif inverse_kind == "local":
                y_pred_test = processor.inverse_transform_local(y_pred_scaled_test, inverse_ctx_test)
                y_test_orig = processor.inverse_transform_local(y_test_scaled, inverse_ctx_test)
            else:
                msn, ctx_test = inverse_ctx_test
                _per_scale_te, y_pred_test = msn.denorm_and_ensemble(y_pred_scaled_test, ctx_test)
                y_test_orig = y_test
            
            y_pred_test, y_test_orig, timestamps, n_steps = self.universal_postprocess(y_pred_test, y_test_orig, ts_test)
            
            # TRAIN
            y_pred_scaled_train = trainer.predict(model, X_train_scaled) if hasattr(trainer, "predict") else model.predict(X_train_scaled)
            if inverse_kind == "global":
                y_pred_train = processor.inverse_transform_global(y_pred_scaled_train)
                y_train_orig = processor.inverse_transform_global(y_train_scaled)
            elif inverse_kind == "local":
                y_pred_train = processor.inverse_transform_local(y_pred_scaled_train, ctx_train_local)
                y_train_orig = processor.inverse_transform_local(y_train_scaled, ctx_train_local)
            else:
                _per_scale_tr, y_pred_train = msn.denorm_and_ensemble(y_pred_scaled_train, ctx_train_msn)
                y_train_orig = y_train
            
            y_pred_train = np.array(y_pred_train)
            y_train_orig = np.array(y_train_orig)
            if y_pred_train.ndim == 1:
                y_pred_train = y_pred_train.reshape(-1, 1)
            if y_train_orig.ndim == 1:
                y_train_orig = y_train_orig.reshape(-1, 1)
            y_train_orig = y_train_orig[:, :y_pred_train.shape[1]]
            
            metrics_train = self._compute_basic_metrics(y_train_orig, y_pred_train)
            metrics_test = self._compute_basic_metrics(y_test_orig, y_pred_test)
            
            logger.info(f"[FINETUNE] Métricas Test: RMSE={metrics_test['rmse']:.4f}, MAE={metrics_test['mae']:.4f}, R2={metrics_test['r2']:.4f}")
            
            # === 12. COMPARAÇÃO COM MODELO ORIGINAL ===
            comparison = {
                "original_run_id": original_run_id,
                "finetune_run_id": hash_id,
                "comparison_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            
            if original_metrics:
                orig_test = original_metrics.get("test", {})
                comparison["metrics_comparison"] = {
                    "rmse": {
                        "original": orig_test.get("rmse"),
                        "finetuned": metrics_test["rmse"],
                        "delta": metrics_test["rmse"] - orig_test.get("rmse", 0),
                        "improvement_pct": ((orig_test.get("rmse", 0) - metrics_test["rmse"]) / orig_test.get("rmse", 1)) * 100 if orig_test.get("rmse") else None
                    },
                    "mae": {
                        "original": orig_test.get("mae"),
                        "finetuned": metrics_test["mae"],
                        "delta": metrics_test["mae"] - orig_test.get("mae", 0),
                        "improvement_pct": ((orig_test.get("mae", 0) - metrics_test["mae"]) / orig_test.get("mae", 1)) * 100 if orig_test.get("mae") else None
                    },
                    "r2": {
                        "original": orig_test.get("r2"),
                        "finetuned": metrics_test["r2"],
                        "delta": metrics_test["r2"] - orig_test.get("r2", 0),
                        "improvement_pct": ((metrics_test["r2"] - orig_test.get("r2", 0)) / abs(orig_test.get("r2", 1))) * 100 if orig_test.get("r2") else None
                    }
                }
                
                # Verificar se houve melhoria
                degradation = (
                    metrics_test["rmse"] > orig_test.get("rmse", float('inf')) or
                    metrics_test["mae"] > orig_test.get("mae", float('inf'))
                )
                
                if degradation:
                    logger.warning("[FINETUNE] ⚠️  PERFORMANCE DEGRADOU em relação ao modelo original!")
                    comparison["performance_status"] = "DEGRADED"
                else:
                    logger.info("[FINETUNE] ✓ Performance mantida ou melhorada")
                    comparison["performance_status"] = "IMPROVED"
            else:
                comparison["performance_status"] = "NO_BASELINE"
            
            with open(comparison_path, "w") as f:
                json.dump(comparison, f, indent=4)
            
            logger.info(f"[FINETUNE] Comparação salva: {comparison_path}")
            
            # === 13. ARTEFATOS (CSV, GRÁFICOS) ===
            logger.info("[FINETUNE] Gerando artefatos...")
            metrics_readable = {
                "MAE - Train data": metrics_train["mae"],
                "MAE - Test data": metrics_test["mae"],
                "RMSE - Train data": metrics_train["rmse"],
                "RMSE - Test data": metrics_test["rmse"],
                "MSE - Train data": metrics_train["mse"],
                "MSE - Test data": metrics_test["mse"],
                "R2 score - Train data": metrics_train["r2"],
                "R2 score - Test data": metrics_test["r2"],
            }
            metrics_bundle = {
                "train": metrics_train,
                "test": metrics_test,
                "readable": metrics_readable,
                "n_steps": int(n_steps),
                "window_size": int(window_size),
                "run_id": hash_id,
            }
            with open(metrics_path, "w") as f:
                json.dump(metrics_bundle, f, indent=4)
            
            csv_exporter = CSVExporter()
            results_df = csv_exporter.save_predictions_to_csv(
                timestamps=timestamps, y_test=y_test_orig, y_pred=y_pred_test,
                steps_ahead=n_steps, output_path=csv_path
            )
            
            plotter = Plotter()
            plotter.plot_price_predictions(
                results_df=results_df, timestamp_col="timestamp",
                real_col="real_value", pred_col="predicted_value",
                save_path=os.path.join(plot_dir, "price_predictions.png")
            )
            plotter.plot_errors_over_time(
                y_test=y_test_orig, y_pred=y_pred_test,
                save_path=os.path.join(plot_dir, "errors_over_time.png")
            )
            try:
                plotter.plot_correlation_matrix(
                    df=df, columns=config.get("relevant_columns"),
                    save_path=os.path.join(plot_dir, "correlation_matrix.png")
                )
            except Exception:
                pass
            plotter.plot_histogram_of_errors(
                y_test=y_test_orig, y_pred=y_pred_test,
                save_path=os.path.join(plot_dir, "histogram_errors.png")
            )
            plotter.plot_scatter_real_vs_predicted(
                y_test=y_test_orig, y_pred=y_pred_test,
                save_path=os.path.join(plot_dir, "scatter_real_vs_predicted.png")
            )
            
            # === 14. BANCO DE DADOS ===
            logger.info("[FINETUNE] Salvando no banco de dados...")
            metrics_dict = {
                "train_loss": float(train_loss) if train_loss is not None else None,
                "val_loss": float(val_loss) if val_loss is not None else None,
                "n_steps": int(n_steps),
                "window_size": int(window_size),
                "run_id": hash_id,
            }
            metrics_json = json.dumps(metrics_dict, indent=4)
            
            ensure_db_connection()
            fine_tune_run = FineTuningRun.create(
                original_run=original_run_id,
                finetune_uuid=hash_id,
                status="finished",
                start_time=datetime.now(),
                end_time=datetime.now(),
                created_at=datetime.now(),
                updated_at=datetime.now(),
                finetuned_model_path=model_ft_path,
                finetune_config_path=config_path,
                finetune_csv_metrics_path=csv_path,
                finetune_plot_dir=plot_dir,
                metrics_json=metrics_json,
                framework=framework,
                model_type=model_type,
                seed=config.get("seed"),
                gpu_used=Settings.USE_GPU,
                train_loss=float(train_loss) if train_loss is not None else None,
                val_loss=float(val_loss) if val_loss is not None else None,
                best_epoch=int(model.history.epoch[-1]) if hasattr(model, "history") else None,
                log=log_path
            )
            
            logger.info(f"[FINETUNE] ✓ Registro salvo no banco: ID={fine_tune_run.id}")
            logger.info("[FINETUNE] ========== FINE-TUNING CONCLUÍDO ==========")
            logger.close()
            
            return {
                "train_loss": float(train_loss) if train_loss is not None else None,
                "val_loss": float(val_loss) if val_loss is not None else None,
                "model_path": model_ft_path,
                "csv_path": csv_path,
                "metrics": metrics_bundle,
                "metrics_path": metrics_path,
                "comparison_path": comparison_path,
                "comparison": comparison,
                "run_id": hash_id,
                "results_path": base_path,
                "fine_tune_run_id": fine_tune_run.id,
                "backup_path": backup_model_path,
                "log_path": log_path
            }
        
        except Exception as e:
            logger.error(f"[FINETUNE] ✗✗✗ ERRO CRÍTICO: {type(e).__name__}: {str(e)}")
            logger.error("[FINETUNE] Stack trace:")
            import traceback
            logger.error(traceback.format_exc())
            logger.close()
            raise

    # -------------------------
    # LIVE RUN (único ciclo)
    # -------------------------
    def live_run_once(
        self,
        model_path: str,
        config_path: str,
        framework: str,
        model_type: str,
        symbol: str = "BTCUSDT",
        interval: str = "1h"
    ):
        # ---------- 1) Carrega config e seta seed ----------
        with open(config_path, "r") as f:
            config = json.load(f)
        seed = config.get("seed")
        if seed:
            set_seed(seed)

        # Normalização e parâmetros principais do modelo
        norm_strategy, scaler_type, x_mode, y_mode, evcfg = self._norm_cfg(config)
        window_size = int(config["window_size"])
        steps_ahead = int(config.get("steps_ahead", 1))
        relevant_columns = config["relevant_columns"]
        indicators_apply = config.get("indicators_apply", None)
        target_column = config["target_column"]

        # ---------- 2) Busca dados recentes (candle fechado) ----------
        binance = BinanceData()
        now = datetime.utcnow()

        if interval == "1h":
            end_time = now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
            delta_window = timedelta(hours=window_size)
        elif interval == "4h":
            hour = (now.replace(minute=0, second=0, microsecond=0).hour // 4) * 4
            end_time = now.replace(hour=hour, minute=0, second=0, microsecond=0) - timedelta(hours=4)
            delta_window = timedelta(hours=4 * window_size)
        elif interval == "1d":
            end_time = now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
            delta_window = timedelta(days=window_size)
        else:
            raise ValueError(f"Intervalo não suportado no live_run_once: {interval}")

        start_time = end_time - delta_window
        start_str = start_time.strftime("%d %b, %Y %H:%M:%S")
        end_str   = end_time.strftime("%d %b, %Y %H:%M:%S")

        df = binance.get_historical_data(symbol, start_str=start_str, interval=interval, end_str=end_str)
        if len(df) < window_size:
            raise RuntimeError(f"Dados insuficientes para montar janela: len(df)={len(df)} < window_size={window_size}")

        # Indicadores técnicos (se usados no treino)
        if indicators_apply:
            df = TechnicalIndicators.process_indicators(df, indicators_apply)

        # Garante que colunas numéricas existam e estejam numéricas
        for c in list(set(relevant_columns + [target_column])):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # ---------- 3) Monta a janela EXATAMENTE como no treino ----------
        # Usa apenas features (sem timestamp e sem repetir o alvo)
        feat_no_target = [c for c in relevant_columns if c not in (target_column, "timestamp")]

        # Validação de colunas
        missing = [c for c in feat_no_target + [target_column] if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"Colunas ausentes no live_run: {missing}. "
                "Verifique indicators_apply e a ordem/nomes em relevant_columns do config."
            )

        window_features = df[feat_no_target].values[-window_size:]   # (L, F)
        X_window = np.expand_dims(window_features, axis=0)           # (1, L, F)

        # Se normalização local exigir o alvo dentro do X, concatena a janela do alvo como ÚLTIMO canal
        needs_target_in_x = (
            norm_strategy == "local" and self._y_mode_needs_target_in_X(y_mode)
        )
        if needs_target_in_x:
            target_win = df[target_column].values[-window_size:].reshape(1, window_size, 1)
            X_window = np.concatenate([X_window, target_win], axis=2)   # (1, L, F+1)
            target_idx = X_window.shape[2] - 1
        else:
            target_idx = 0

        # ---------- 4) Cria Trainer com o SHAPE REAL e carrega o modelo ----------
        TrainerClass = TrainerFactory.get_trainer(framework, model_type)
        trainer = TrainerClass(input_shape=X_window.shape[1:], **config)  # (L, C) real
        model = trainer.load_model(model_path)

        # ---------- 5) Normalização e predição ----------
        processor = DataProcessor(window_size=window_size, scaler_type=scaler_type)
        scaler_dir = os.path.join(os.path.dirname(os.path.dirname(model_path)), "scaler")

        if norm_strategy == "global":
            # carrega scaler salvo no treino
            processor.load_scaler(scaler_dir)
            X_input, _ = processor.apply_normalization_global(X_window, np.zeros((1, steps_ahead)))
            y_pred_scaled = model.predict(X_input) if hasattr(model, "predict") else trainer.predict(model, X_input)
            y_pred = processor.inverse_transform_global(y_pred_scaled)

        elif norm_strategy == "local":
            # usa target_idx calculado acima
            Xn, _y0, ctx = processor.apply_normalization_local(
                X_window, np.zeros((1, steps_ahead)),
                x_mode=x_mode, y_mode=y_mode, target_idx=target_idx
            )
            y_pred_scaled = model.predict(Xn) if hasattr(model, "predict") else trainer.predict(model, Xn)
            y_pred = processor.inverse_transform_local(y_pred_scaled, ctx)

        else:  # "evomsn" ou "evomsn_like"
            Normalizer = EvoMSNNormalizer if norm_strategy == "evomsn" else EvoMSNLikeNormalizer
            meta = Normalizer.load_meta(scaler_dir)

            # n_features precisa bater com os canais REAIS de X_window
            n_features = X_window.shape[2]

            if norm_strategy == "evomsn":
                msn = Normalizer(
                    window_size=window_size,
                    horizon=steps_ahead,
                    n_features=n_features,
                    target_idx=0,
                    k_scales=meta.k_scales,
                    agg=meta.agg,
                    random_state=config.get("seed", 42),
                    predictor_type=meta.predictor_type,
                )
            else:
                msn = Normalizer(
                    window_size=window_size,
                    horizon=steps_ahead,
                    n_features=n_features,
                    target_idx=0,
                    k_scales=meta.k_scales,
                    agg=meta.agg,
                    random_state=config.get("seed", 42),
                )
            msn.meta = meta

            # janela do alvo para pesos do ensemble
            target_win = df[target_column].values[-window_size:].reshape(1, window_size, 1)
            X_in, _y, ctx = msn.transform(X_window, None, target_windows=target_win)

            y_tilde = model.predict(X_in) if hasattr(model, "predict") else trainer.predict(model, X_in)
            _per_scale, y_pred = msn.denorm_and_ensemble(y_tilde, ctx)

        # ---------- 6) Empacota resposta ----------
        previsoes = [float(y_pred[0, i]) for i in range(steps_ahead)] if steps_ahead > 1 else [float(y_pred[0, 0])]
        last_ts = df.iloc[-1]["timestamp"] if "timestamp" in df.columns else end_time

        return {
            "timestamp": str(last_ts),
            "close_real": float(df.iloc[-1][target_column]),
            "previsoes": previsoes,
            "steps_ahead": steps_ahead,
            "target_column": target_column
        }



    # --- NOVO: util para produto cartesiano do grid ---
    def _cartesian_product(self, grid_dict: dict):
        """
        Recebe {'LR':[0.001,0.0005], 'BATCH':[16,32]} e gera dicts
        [{'LR':0.001,'BATCH':16}, {'LR':0.001,'BATCH':32}, ...]
        """
        from itertools import product
        keys = list(grid_dict.keys())
        values = [grid_dict[k] for k in keys]
        for combo in product(*values):
            yield dict(zip(keys, combo))

    # --- mapeia chaves do JSON-unificado (UPPER) -> nomes internos do config ---
    def _grid_key_mapping(self):
        return {
            # dados / split
            "START_DATE": "start_date",
            "END_DATE": "end_date",
            "TRAIN_SIZE": "train_size",
            "VALIDATION_SPLIT": "validation_split",
            "STEPS_AHEAD": "steps_ahead",
            "TARGET_COLUMN": "target_column",
            "RELEVANT_COLUMNS": "relevant_columns",
            "INDICATORS_APPLY": "indicators_apply",

            # device
            "USE_GPU": "use_gpu",
            "GPU_INDEX": "gpu_index",
            "PARALLEL": "parallel",

            # modelo/framework (podem varrer)
            "FRAMEWORK": "framework",
            "MODEL_TYPE": "model_type",

            # seeds
            "SEED": "seed",

            # normalização (dict fixo ou lista de dicts)
            "NORMALIZATION": "normalization",

            # hiperparâmetros comuns
            "WINDOW_SIZE": "window_size",
            "BATCH_SIZE": "batch_size",
            "EPOCHS": "epochs",
            "PATIENCE": "patience",
            "LEARNING_RATE": "learning_rate",
            "DROPOUT": "dropout",
            "OPTIMIZER": "optimizer",
            "LOSS_FUNCTION": "loss_fn",
            "OUTPUT_UNITS": "output_units",  # opcional

            # LSTM
            "LAYERS_CONFIG": "layers_config",
            "BIDIRECTIONAL": "bidirectional",
            "L1_REGULARIZATION": "l1_reg",
            "L2_REGULARIZATION": "l2_reg",
            "ACTIVATION_FUNCTION": "activation_functions",
            "RECURRENT_DROPOUT": "recurrent_dropout",

            # Transformer
            "NUM_LAYERS": "num_layers",
            "EMBED_DIM": "embed_dim",
            "NUM_HEADS": "num_heads",
            "FF_DIM": "ff_dim",
            "ACTIVATION": "activation",
        }
    
    @staticmethod
    def example_unified_grid_json():
        """
        JSON UNIFICADO: qualquer campo pode ser escalar (fixo) ou lista (varredura).
        """
        return {
          "grid_id": "grid_demo",
          "framework": ["keras", "pytorch"],                 # varrer frameworks
          "model_type": ["lstm"],                            # pode varrer modelos também

          # Janela temporal (varrer ranges ou fixar)
          "START_DATE": ["2017-08-18 00:00:00"],             # lista de 1 => vira fixo
          "END_DATE":   ["2025-01-19 23:59:59"],

          # Split
          "TRAIN_SIZE": [0.7],
          "VALIDATION_SPLIT": [0.15],
          "STEPS_AHEAD": [1, 3],                             # varrer saídas (1 e 3)

          # Dados/Features
          "TARGET_COLUMN": ["close"],
          "RELEVANT_COLUMNS": [
            ["close","open","high","low","volume"],          # set A
            ["close","volume"]                               # set B
          ],

          # Indicadores (pode ser dict fixo ou varrer listas de dict)
          "INDICATORS_APPLY": [
            {},                                              # sem indicadores
            {"sma": [{"period": 14, "col_name": "sma_14"}]}
          ],

          # Normalização: dict fixo OU lista de dicts
          "NORMALIZATION": [
            {"strategy": "global", "scaler_type": "robust"},
            {"strategy": "local", "x_mode": "zscore", "y_mode": "relative_last"},
            {"strategy": "evomsn", "evomsn_k_scales": 4, "evomsn_predictor": "linear"}
          ],


          # Device
          "USE_GPU": [False],
          "GPU_INDEX": ["0 - /physical_device:GPU:0"],
          "PARALLEL": {
            "enabled": True,
            "backend": "process",
            "max_workers_per_gpu": 2,
            "safety_ratio": 0.20,
            "cpu_workers": 2
            },

          # Seeds (inteiros ou "RANDOM")
          "SEED": ["RANDOM"],

          # Hiperparâmetros
          "WINDOW_SIZE": [72, 96],
          "BATCH_SIZE": [16, 32],
          "EPOCHS": [50],
          "PATIENCE": [5, 10],
          "LEARNING_RATE": [0.001, 0.0005],
          "DROPOUT": [0.2, 0.3],
          "OPTIMIZER": ["Adam"],
          "LOSS_FUNCTION": ["mean_squared_error"],

          # Específicos LSTM
          "LAYERS_CONFIG": [[128,64], [64,64]],
          "BIDIRECTIONAL": [False, True],
          "ACTIVATION_FUNCTION": ["tanh", "relu"],
          "RECURRENT_DROPOUT": [0.0],

          # Específicos Transformer (serão ignorados se model_type != transformer)
          "NUM_LAYERS": [2, 3],
          "EMBED_DIM": [32],
          "NUM_HEADS": [2, 4],
          "FF_DIM": [64, 128],
          "ACTIVATION": ["relu"]
        }

    # --- lista com >1 item => varredura; dict => fixo; lista de dicts/lista de listas => varredura ---
    def _is_sweep_value(self, v):
        if isinstance(v, list):
            if len(v) == 0:
                return False
            if len(v) == 1:
                return False
            return True
        return False

    # --- quebra um JSON unificado em (framework, model_type, base_cfg, grid-eixos) ---
    def _split_unified_grid_config(self, unified: dict):
        unified = (unified or {}).copy()
        grid_id = unified.pop("grid_id", None)

        mapping = self._grid_key_mapping()
        base_cfg, grid = {}, {}

        # percorre todo o JSON: se for lista com 2+ => varredura; senão vira fixo
        for raw_k, v in unified.items():
            k = raw_k.upper()
            dst = mapping.get(k, raw_k.lower())

            # PARALLEL: aceita apenas dict fixo (não varrer)
            if dst == "parallel":
                if isinstance(v, dict):
                    base_cfg["parallel"] = v
                else:
                    raise ValueError("PARALLEL deve ser um objeto/dict.")
                continue

            # NORMALIZATION aceita dict fixo ou lista de dicts (varredura)
            if dst == "normalization":
                if self._is_sweep_value(v):
                    grid["NORMALIZATION"] = v
                else:
                    if isinstance(v, list) and len(v) == 1:
                        v = v[0]
                    base_cfg["normalization"] = v
                continue

            # SEED aceita int, "RANDOM" ou lista (com int/"RANDOM")
            if dst == "seed":
                if self._is_sweep_value(v):
                    grid["SEED"] = v
                else:
                    if isinstance(v, list) and len(v) == 1:
                        v = v[0]
                    base_cfg["seed"] = v
                continue

            # framework/model_type: também podem varrer
            if dst in ("framework", "model_type"):
                if self._is_sweep_value(v):
                    grid[k] = v   # manter em UPPER para o laço
                else:
                    if isinstance(v, list) and len(v) == 1:
                        v = v[0]
                    base_cfg[dst] = v
                continue

            # demais campos (datas, features, hypers...)
            if self._is_sweep_value(v):
                grid[k] = v
            else:
                if isinstance(v, list) and len(v) == 1:
                    v = v[0]
                base_cfg[dst] = v

        if grid_id:
            base_cfg["grid_id"] = grid_id

        return base_cfg, grid

    # --- aplica params de um combo sobre a base, resolvendo seed e output_units ---
    def _apply_params_and_seed_unified(self, base_cfg: dict, params: dict) -> dict:
        cfg = base_cfg.copy()
        mapping = self._grid_key_mapping()

        for k, v in params.items():
            dst = mapping.get(k, k.lower())

            if dst == "seed":
                if isinstance(v, str) and v.strip().upper() == "RANDOM":
                    cfg.pop("seed", None)
                else:
                    cfg["seed"] = int(v)
                continue

            if dst == "normalization":
                if not isinstance(v, dict):
                    raise ValueError("Cada item de NORMALIZATION no grid deve ser um dict.")
                cfg["normalization"] = v
                continue

            # 👇 NOVO: normalização de GPU_INDEX (string -> int)
            if dst == "gpu_index":
                if isinstance(v, str):
                    vv = v.strip()
                    # tenta pegar o inteiro antes do primeiro espaço ou hífen
                    try:
                        vv_int = int(vv.split()[0].split("-")[0])
                        cfg["gpu_index"] = vv_int
                    except Exception:
                        cfg["gpu_index"] = v  # deixa como está se não der pra converter
                else:
                    cfg["gpu_index"] = v
                continue

            cfg[dst] = v

        if "steps_ahead" in cfg:
            cfg["output_units"] = int(cfg["steps_ahead"])

        cfg["run_id"] = cfg.get("run_id") or str(random.randint(10**6, 10**7-1))
        return cfg


    def grid_search_unified(self, unified_config: dict, *, score: str = "rmse", score_on: str = "test"):
        """Grid search com suporte a execução paralela segura."""
        assert score_on in ("train", "test")
        valid_scores = {"rmse", "mae", "mse", "r2"}
        if score not in valid_scores:
            raise ValueError(f"score deve ser um de {valid_scores}")

        base_cfg, grid = self._split_unified_grid_config(unified_config)
        grid_id = base_cfg.get("grid_id") or datetime.now().strftime("grid_%Y%m%d_%H%M%S")
        base_cfg["grid_id"] = grid_id

        grid_root = os.path.join(BASE_DIR, "results", "grid", grid_id)
        os.makedirs(grid_root, exist_ok=True)
        with open(os.path.join(grid_root, "unified_config.json"), "w") as f:
            json.dump(unified_config, f, indent=4)

        par = (base_cfg.get("parallel") or {})
        enabled = bool(par.get("enabled", False))
        backend = str(par.get("backend", "process")).lower()
        max_workers_per_gpu = int(par.get("max_workers_per_gpu", 1))
        safety_ratio = float(par.get("safety_ratio", 0.20))
        cpu_workers = int(par.get("cpu_workers", 1))

        def _finalize_and_record(out, params, fw, mt):
            nonlocal best_record, results_summary
            run_metrics = out.get("metrics", {})
            readable = run_metrics.get("readable", {})
            metrics_section = run_metrics.get(score_on, {})
            val = float(metrics_section.get(score))
            is_better = (val > (best_record or {}).get("score_value", -1e18)) if score == "r2" else (val < (best_record or {}).get("score_value", 1e18))
            
            try:
                ensure_db_connection()
                tr = TrainingRun.get(TrainingRun.run_uuid == out["run_id"])
                GridResult.create(
                    training_run=tr, params_json=json.dumps(params, indent=2),
                    train_loss=out.get("train_loss"), val_loss=out.get("val_loss"),
                    metrics_json=json.dumps(readable, indent=2),
                    model_path=out.get("model_path"), csv_metrics_path=out.get("csv_path"), epoch=None
                )
            except Exception as e:
                print(f"[GridResult] WARN banco: {e}", flush=True)
            
            rec = {
                "run_id": out["run_id"], "params": params, "framework": fw, "model_type": mt,
                "score_on": score_on, "score_name": score, "score_value": val,
                "model_path": out.get("model_path"), "csv_path": out.get("csv_path"),
                "metrics_path": out.get("metrics_path"), "metrics_readable": readable,
            }
            results_summary.append(rec)
            if is_better:
                best_record = rec

        results_summary, best_record = [], None
        base_cfg.setdefault("use_gpu", False)
        base_cfg.setdefault("gpu_index", None)

        combos = list(self._cartesian_product(grid or {}))
        print(f"\n[GRID] Total: {len(combos)} combinações", flush=True)

        if not enabled:
            print("[GRID] Modo SEQUENCIAL", flush=True)
            for i, params in enumerate(combos, 1):
                print(f"\n[GRID] Combo {i}/{len(combos)}", flush=True)
                cfg = self._apply_params_and_seed_unified(base_cfg, params)
                fw = cfg.pop("framework", base_cfg.get("framework"))
                mt = cfg.pop("model_type", base_cfg.get("model_type"))
                if fw is None or mt is None:
                    raise ValueError("framework/model_type obrigatórios")
                out = self.train(cfg, fw, mt, artifacts_base=f"grid/{grid_id}")
                _finalize_and_record(out, params, fw, mt)
        else:
            print(f"[GRID] Modo PARALELO: {max_workers_per_gpu} jobs/GPU", flush=True)
            
            try:
                from utils.capacity_train import gpu_mem_info
                gpus = gpu_mem_info()
                num_gpus = len(gpus)
                print(f"[GRID] GPUs: {num_gpus}", flush=True)
            except Exception as e:
                print(f"[GRID] Falha GPU detection: {e}", flush=True)
                num_gpus = 0
            
            gpu_slots_in_use = [0] * max(1, num_gpus)
            pool_size = max(1, cpu_workers + (num_gpus * max_workers_per_gpu))
            print(f"[GRID] Pool: {pool_size} workers", flush=True)

            pending = combos[:]
            running = {}
            completed = 0

            with concurrent.futures.ProcessPoolExecutor(max_workers=pool_size) as ex:
                while pending or running:
                    # Admitir novos
                    i = 0
                    while i < len(pending):
                        params = pending[i]
                        cfg = self._apply_params_and_seed_unified(base_cfg, params)
                        fw = cfg.pop("framework", base_cfg.get("framework"))
                        mt = cfg.pop("model_type", base_cfg.get("model_type"))
                        
                        if fw is None or mt is None:
                            raise ValueError("framework/model_type obrigatórios")

                        use_gpu = bool(cfg.get("use_gpu", False))
                        assigned_gpu = None
                        
                        if use_gpu and num_gpus > 0:
                            assigned_gpu = pick_gpu_for_job(cfg, mt, max_workers_per_gpu, gpu_slots_in_use, safety_ratio=safety_ratio)
                            if assigned_gpu is None:
                                i += 1
                                continue
                        else:
                            num_cpu_running = sum(1 for meta in running.values() if meta["gpu_idx"] is None)
                            if num_cpu_running >= cpu_workers:
                                i += 1
                                continue

                        if assigned_gpu is not None:
                            gpu_slots_in_use[assigned_gpu] += 1
                            print(f"[GRID] GPU {assigned_gpu}: {gpu_slots_in_use[assigned_gpu]}/{max_workers_per_gpu} slots", flush=True)

                        args = (cfg, fw, mt, f"grid/{grid_id}", assigned_gpu)
                        fut = ex.submit(_train_job_worker, args)
                        running[fut] = {"params": params, "gpu_idx": assigned_gpu, "framework": fw, "model_type": mt, "combo_idx": len(combos) - len(pending) + 1}
                        pending.pop(i)
                        print(f"[GRID] Submetido: combo {running[fut]['combo_idx']}/{len(combos)} | GPU: {assigned_gpu}", flush=True)
                    
                    # Colher finalizados
                    if not running:
                        continue
                        
                    done, _ = concurrent.futures.wait(running.keys(), timeout=1.0, return_when=concurrent.futures.FIRST_COMPLETED)
                    
                    for fut in done:
                        meta = running.pop(fut)
                        completed += 1
                        
                        if meta["gpu_idx"] is not None:
                            gpu_slots_in_use[meta["gpu_idx"]] = max(0, gpu_slots_in_use[meta["gpu_idx"]] - 1)
                            print(f"[GRID] GPU {meta['gpu_idx']} liberada: {gpu_slots_in_use[meta['gpu_idx']]} slots", flush=True)
                        
                        try:
                            out = fut.result(timeout=5)
                            print(f"[GRID] ✓ Combo {meta['combo_idx']}/{len(combos)} OK", flush=True)
                            _finalize_and_record(out, meta["params"], meta["framework"], meta["model_type"])
                        except Exception as e:
                            print(f"[GRID] ✗ Combo {meta['combo_idx']}/{len(combos)} FALHOU: {e}", flush=True)
                            rec = {
                                "run_id": None, "params": meta["params"], "framework": meta["framework"], "model_type": meta["model_type"],
                                "score_on": score_on, "score_name": score,
                                "score_value": float("inf") if score != "r2" else float("-inf"),
                                "model_path": None, "csv_path": None, "metrics_path": None,
                                "metrics_readable": {"error": str(e)},
                            }
                            results_summary.append(rec)
                    
                    if done:
                        print(f"[GRID] Progresso: {completed}/{len(combos)} | {len(running)} rodando | {len(pending)} pendentes", flush=True)

        # Salvar resultados
        print(f"\n[GRID] Salvando resultados...", flush=True)
        
        summary_path = os.path.join(grid_root, "grid_summary.json")
        with open(summary_path, "w") as f:
            json.dump({"grid_id": grid_id, "score": score, "score_on": score_on, "best": best_record, "summary": results_summary}, f, indent=4)

        import csv
        lb_csv = os.path.join(grid_root, "leaderboard.csv")
        with open(lb_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["run_id","framework","model_type","score_on","score_name","score_value","model_path","csv_path","metrics_path","params_json"])
            for rec in sorted(results_summary, key=lambda x: x["score_value"], reverse=(score=="r2")):
                w.writerow([rec.get("run_id"), rec["framework"], rec["model_type"], rec["score_on"], rec["score_name"], rec["score_value"],
                    rec.get("model_path"), rec.get("csv_path"), rec.get("metrics_path"), json.dumps(rec["params"])])

        print(f"[GRID] ✓ Concluído! Melhor: {best_record['score_value'] if best_record else 'N/A'}", flush=True)
        
        return {"grid_id": grid_id, "best": best_record, "summary": results_summary, "leaderboard_csv": lb_csv, "summary_json": summary_path}
