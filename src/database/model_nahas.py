from peewee import FloatField, DateTimeField, AutoField, CharField, TextField, ForeignKeyField, IntegerField
from datetime import datetime
from database.model_base import BaseModel, db


class Experiment(BaseModel):
    id = AutoField()
    type = CharField()  # 'grid', 'single', 'finetune', etc.
    framework = CharField()  # 'keras', 'pytorch', etc.
    model_type = CharField()  # 'lstm', 'transformer', etc.
    hyperparams_json = TextField()  # JSON string com hiperparâmetros do experimento (ou JSONField)
    hash_config = CharField(unique=True)
    user = CharField(null=True)
    status = CharField(default="created")  # 'created', 'running', 'finished', etc.
    notes = TextField(null=True)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

class TrainingRun(BaseModel):
    id = AutoField()
    experiment = ForeignKeyField(Experiment, backref="runs")
    run_uuid = CharField(unique=True)  # uuid4
    start_time = DateTimeField(default=datetime.now)
    end_time = DateTimeField(null=True)
    status = CharField(default="started")  # 'started', 'finished', 'failed', etc.
    model_path = CharField(null=True)
    csv_metrics_path = CharField(null=True)
    metrics_json = TextField(null=True)  # JSON string com métricas
    best_epoch = IntegerField(null=True)
    log = TextField(null=True)
    created_at = DateTimeField(default=datetime.now)
    updated_at = DateTimeField(default=datetime.now)

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

# Crie as tabelas no banco de dados
db.connect()
db.create_tables([Experiment, TrainingRun, GridResult], safe=True)
db.close()