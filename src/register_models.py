"""Cadastra no banco (TrainingRun) os modelos que já existem em results/train/<hash>/.

Útil quando os artefatos vieram de outra máquina/banco: a interface web só lista
o que está na tabela TrainingRun. Idempotente — pula hashes já cadastrados.

Uso (de dentro de src/):
    python register_models.py
"""
import glob
import json
import os
import zipfile
from datetime import datetime

from database.init_db import init_db
from database.model_base import db
from database.model_nahas import TrainingRun

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "train")


def detect_model_type(model_path: str) -> str:
    with zipfile.ZipFile(model_path) as z:
        cfg = z.read("config.json").decode()
    return "transformer" if "MultiHeadAttention" in cfg else "lstm"


def main():
    init_db()
    db.connect(reuse_if_open=True)
    for run_dir in sorted(glob.glob(os.path.join(BASE, "*"))):
        run_id = os.path.basename(run_dir)
        models = glob.glob(os.path.join(run_dir, "models", "*.keras"))
        configs = glob.glob(os.path.join(run_dir, "hiperparams", "*.json"))
        if not models or not configs:
            print(f"[pula] {run_id}: sem .keras ou config")
            continue
        if TrainingRun.select().where(TrainingRun.run_uuid == run_id).exists():
            print(f"[ok]   {run_id}: já cadastrado")
            continue

        with open(configs[0]) as f:
            cfg = json.load(f)
        # Treinados antes do commit cef7699: política antiga do canal do alvo.
        # Sem a chave, o código atual assume "equalized" e o nº de features diverge.
        if "include_target_channel" not in cfg:
            cfg["include_target_channel"] = "legacy"
            with open(configs[0], "w") as f:
                json.dump(cfg, f, indent=4)

        created_at = datetime.now()
        meta_path = os.path.join(run_dir, "model_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                created_at = datetime.strptime(json.load(f)["created_at"], "%Y-%m-%d %H:%M:%S")

        metrics_path = os.path.join(run_dir, "metrics.json")
        csvs = glob.glob(os.path.join(run_dir, "csv", "*.csv"))
        model_type = detect_model_type(models[0])
        TrainingRun.create(
            run_uuid=run_id,
            status="finished",
            start_time=created_at,
            end_time=created_at,
            created_at=created_at,
            updated_at=datetime.now(),
            model_path=models[0],
            config_path=configs[0],
            csv_metrics_path=csvs[0] if csvs else None,
            plot_dir=os.path.join(run_dir, "graficos"),
            metrics_json=open(metrics_path).read() if os.path.exists(metrics_path) else None,
            framework="keras",
            model_type=model_type,
            target_column=cfg.get("target_column"),
            seed=cfg.get("seed"),
            gpu_used=cfg.get("use_gpu"),
            log=f"importado de results/train ({cfg.get('symbol')} {cfg.get('interval')})",
        )
        print(f"[novo] {run_id}: {cfg.get('symbol')} keras/{model_type}")
    db.close()


if __name__ == "__main__":
    main()
