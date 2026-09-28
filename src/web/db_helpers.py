from peewee import InternalError, OperationalError
from database.model_nahas import TrainingRun
from utils.db_utils import ensure_db_connection
from database.model_base import db

def list_all_models(framework=None, model_type=None, status="finished"):
    def _query_once():
        q = TrainingRun.select()
        if framework:
            q = q.where(TrainingRun.framework == framework)
        if model_type:
            q = q.where(TrainingRun.model_type == model_type)
        if status:
            q = q.where(TrainingRun.status == status)
        q = q.order_by(TrainingRun.created_at.desc())
        return [
            {
                "id": run.id,
                "run_uuid": run.run_uuid,
                "framework": run.framework,
                "model_type": run.model_type,
                "target_column": run.target_column,
                "created_at": run.created_at,
                "model_path": run.model_path,
                "config_path": run.config_path,
                "csv_metrics_path": run.csv_metrics_path,
                "plot_dir": run.plot_dir,
                "seed": run.seed,
                "gpu_used": run.gpu_used,
                "train_loss": run.train_loss,
                "val_loss": run.val_loss,
                "status": run.status,
                "metrics_json": run.metrics_json,
                "notes": run.log,
            }
            for run in q
        ]

    # 1) garante conexão saudável
    ensure_db_connection()
    try:
        return _query_once()
    except (InternalError, OperationalError):
        # 2) conexão pode ter ficado inválida (comum após ProcessPool)
        try:
            if not db.is_closed():
                db.close()
        except Exception:
            pass
        db.connect(reuse_if_open=True)
        # 3) tenta novamente
        return _query_once()
