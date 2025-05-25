from config.settings import MODEL_TYPE_CLASSNAME, FRAMEWORK_CLASSNAME


class TrainerFactory:
    @staticmethod
    def get_trainer(framework: str, model_type: str):
        framework_name = FRAMEWORK_CLASSNAME.get(framework.lower())
        if not framework_name:
            raise ValueError(f"Framework '{framework}' não suportado.")
        model_type_name = MODEL_TYPE_CLASSNAME.get(model_type.lower())
        if not model_type_name:
            raise ValueError(f"Tipo de modelo '{model_type}' não suportado.")

        class_name = f"{framework_name}{model_type_name}Trainer"
        module_path = f"models.{framework.lower()}.{model_type.lower()}_trainer"

        try:
            exec(f"from {module_path} import {class_name}", globals())
            trainer_class = eval(class_name)
            return trainer_class
        except Exception as e:
            raise ImportError(
                f"Trainer não encontrado: from {module_path} import {class_name}\nErro: {e}"
            )