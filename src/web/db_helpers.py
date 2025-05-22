
from database.model_nahas import TrainingRun, Experiment

def list_all_models(framework=None, model_type=None, user=None, status="finished"):
    query = Experiment.select().join(TrainingRun)
    if framework:
        query = query.where(Experiment.framework == framework)
    if model_type:
        query = query.where(Experiment.model_type == model_type)
    if status:
        query = query.where(Experiment.status == status)
    if user:
        query = query.where(Experiment.user == user)
    query = query.order_by(Experiment.created_at.desc())
    # Pode customizar o retorno aqui, exemplo:
    return [
        {
            "id": exp.id,
            "framework": exp.framework,
            "model_type": exp.model_type,
            "hash_config": exp.hash_config,
            "created_at": exp.created_at,
            "model_path": exp.runs[0].model_path if exp.runs else "",
            "notes": exp.notes,
        }
        for exp in query
    ]
