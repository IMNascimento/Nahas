import re

class ValidationError(Exception):
    """Exceção personalizada para erros de validação."""
    def __init__(self, message):
        super().__init__(message)

class DataValidator:
    """Classe profissional para validação de diferentes tipos de dados."""
    
    @staticmethod
    def validate_string(value, min_length=1, max_length=None, allow_empty=False):
        """Valida se o valor é uma string válida."""
        if not isinstance(value, str):
            raise ValidationError(f"Esperado uma string, mas recebido {type(value).__name__}")
        
        if not allow_empty and not value:
            raise ValidationError("A string não pode ser vazia.")
        
        if min_length and len(value) < min_length:
            raise ValidationError(f"A string deve ter pelo menos {min_length} caracteres.")
        
        if max_length and len(value) > max_length:
            raise ValidationError(f"A string deve ter no máximo {max_length} caracteres.")
        
        return True

    @staticmethod
    def validate_integer(value, min_value=None, max_value=None):
        """Valida se o valor é um número inteiro válido."""
        if not isinstance(value, int):
            raise ValidationError(f"Esperado um número inteiro, mas recebido {type(value).__name__}")
        
        if min_value is not None and value < min_value:
            raise ValidationError(f"O valor deve ser maior ou igual a {min_value}.")
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"O valor deve ser menor ou igual a {max_value}.")
        
        return True

    @staticmethod
    def validate_float(value, min_value=None, max_value=None):
        """Valida se o valor é um número float válido."""
        if not isinstance(value, float):
            raise ValidationError(f"Esperado um número float, mas recebido {type(value).__name__}")
        
        if min_value is not None and value < min_value:
            raise ValidationError(f"O valor deve ser maior ou igual a {min_value}.")
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"O valor deve ser menor ou igual a {max_value}.")
        
        return True

    @staticmethod
    def validate_email(value):
        """Valida se o valor é um endereço de e-mail válido."""
        if not isinstance(value, str):
            raise ValidationError(f"Esperado uma string para o e-mail, mas recebido {type(value).__name__}")
        
        email_regex = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"
        if not re.match(email_regex, value):
            raise ValidationError(f"O e-mail '{value}' é inválido.")
        
        return True

    @staticmethod
    def validate_list(value, item_type=None, min_length=1, max_length=None):
        """Valida se o valor é uma lista válida e opcionalmente valida os tipos dos itens."""
        if not isinstance(value, list):
            raise ValidationError(f"Esperado uma lista, mas recebido {type(value).__name__}")
        
        if len(value) < min_length:
            raise ValidationError(f"A lista deve ter pelo menos {min_length} itens.")
        
        if max_length and len(value) > max_length:
            raise ValidationError(f"A lista deve ter no máximo {max_length} itens.")
        
        if item_type:
            for item in value:
                if not isinstance(item, item_type):
                    raise ValidationError(f"Todos os itens da lista devem ser do tipo {item_type.__name__}")
        
        return True
    
    @staticmethod
    def validate_dict(value, key_type=None, value_type=None, min_length=1, max_length=None):
        """Valida se o valor é um dicionário válido e opcionalmente valida os tipos de chaves e valores."""
        if not isinstance(value, dict):
            raise ValidationError(f"Esperado um dicionário, mas recebido {type(value).__name__}")

        if len(value) < min_length:
            raise ValidationError(f"O dicionário deve ter pelo menos {min_length} itens.")
        
        if max_length and len(value) > max_length:
            raise ValidationError(f"O dicionário deve ter no máximo {max_length} itens.")

        if key_type or value_type:
            for key, val in value.items():
                if key_type and not isinstance(key, key_type):
                    raise ValidationError(f"Todas as chaves do dicionário devem ser do tipo {key_type.__name__}, mas '{key}' é do tipo {type(key).__name__}.")
                
                if value_type and not isinstance(val, value_type):
                    raise ValidationError(f"Todos os valores do dicionário devem ser do tipo {value_type.__name__}, mas o valor '{val}' é do tipo {type(val).__name__}.")

        return True