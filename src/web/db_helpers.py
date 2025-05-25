from database.model_nahas import TrainingRun

def list_all_models(framework=None, model_type=None, status="finished"):
    query = TrainingRun.select()
    if framework:
        query = query.where(TrainingRun.framework == framework)
    if model_type:
        query = query.where(TrainingRun.model_type == model_type)
    if status:
        query = query.where(TrainingRun.status == status)
    query = query.order_by(TrainingRun.created_at.desc())
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
        for run in query
    ]