import importlib

class TrainerFactory:
    @staticmethod
    def get_trainer(framework: str, model_type: str):
        module_path = f"models.{framework.lower()}.{model_type.lower()}_trainer"
        class_name = f"{framework.capitalize()}{model_type.capitalize()}Trainer"
        try:
            module = importlib.import_module(module_path)
            trainer_class = getattr(module, class_name)
            return trainer_class
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Trainer não encontrado: {module_path}.{class_name}\nErro: {e}")