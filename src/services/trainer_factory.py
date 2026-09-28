import importlib

from config.settings import MODEL_TYPE_CLASSNAME, FRAMEWORK_CLASSNAME


class TrainerFactory:
    @staticmethod
    def get_trainer(framework: str, model_type: str):
        """Resolve a classe de treinador a partir do framework e do tipo de modelo.

        A importação é feita com importlib, e não com exec/eval sobre string: além
        de mais legível, não liga o interpretador a um texto montado em tempo de
        execução, que é o padrão que qualquer verificador de segurança aponta.
        """
        framework_name = FRAMEWORK_CLASSNAME.get(framework.lower())
        if not framework_name:
            raise ValueError(f"Framework '{framework}' não suportado.")
        model_type_name = MODEL_TYPE_CLASSNAME.get(model_type.lower())
        if not model_type_name:
            raise ValueError(f"Tipo de modelo '{model_type}' não suportado.")

        class_name = f"{framework_name}{model_type_name}Trainer"
        module_path = f"models.{framework.lower()}.{model_type.lower()}_trainer"

        try:
            modulo = importlib.import_module(module_path)
            return getattr(modulo, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(
                f"Trainer não encontrado: from {module_path} import {class_name}\n"
                f"Erro: {e}\n"
                f"Se o framework for pytorch, instale os extras: "
                f"pip install -r requirements-optional.txt"
            )
