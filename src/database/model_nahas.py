from peewee import FloatField, DateTimeField, AutoField, CharField, TextField, ForeignKeyField, IntegerField, BooleanField
from datetime import datetime
from database.model_base import BaseModel


class TrainingRun(BaseModel):
    id = AutoField()
    run_uuid = CharField(unique=True)   # uuid4
    status = CharField(default="started")
    start_time = DateTimeField(default=datetime.now)
    end_time = DateTimeField(null=True)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)
    model_path = CharField(null=True)
    csv_metrics_path = CharField(null=True)
    plot_dir = CharField(null=True)
    config_path = CharField(null=True)  # Caminho para o JSON de config/hyperparams
    metrics_json = TextField(null=True) # Principais métricas (json string)
    framework = CharField(null=True)
    model_type = CharField(null=True)
    target_column = CharField(null=True)
    seed = IntegerField(null=True)
    gpu_used = BooleanField(null=True)
    train_loss = FloatField(null=True)
    val_loss = FloatField(null=True)
    best_epoch = IntegerField(null=True)
    log = TextField(null=True)


class FineTuningRun(BaseModel):
    id = AutoField()
    original_run = ForeignKeyField(TrainingRun, backref="finetunes", to_field="run_uuid")  # Relaciona com o treino base
    finetune_uuid = CharField(unique=True)  # uuid4
    status = CharField(default="started")  # started, finished, failed, etc.
    start_time = DateTimeField(default=datetime.now)
    end_time = DateTimeField(null=True)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)
    # Novos arquivos e paths
    finetuned_model_path = CharField(null=True)      # Caminho do novo modelo salvo
    finetune_config_path = CharField(null=True)      # Caminho do config/hyperparams usado no fine-tune (json)
    finetune_csv_metrics_path = CharField(null=True) # Caminho dos resultados CSV
    finetune_plot_dir = CharField(null=True)         # Pasta de gráficos desta run de FT
    metrics_json = TextField(null=True)              # Principais métricas (json string)
    # Meta-infos relevantes do fine-tune
    framework = CharField(null=True)
    model_type = CharField(null=True)
    seed = IntegerField(null=True)
    gpu_used = BooleanField(null=True)
    train_loss = FloatField(null=True)
    val_loss = FloatField(null=True)
    best_epoch = IntegerField(null=True)
    log = TextField(null=True)                      

class GridResult(BaseModel):
    id = AutoField()
    training_run = ForeignKeyField(TrainingRun, backref="grid_results")
    params_json = TextField()  # JSON dos hiperparâmetros específicos desse teste do grid
    train_loss = FloatField(null=True)
    val_loss = FloatField(null=True)
    metrics_json = TextField(null=True)  # Todas métricas extras, se quiser
    model_path = CharField(null=True)
    csv_metrics_path = CharField(null=True)
    epoch = IntegerField(null=True)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

