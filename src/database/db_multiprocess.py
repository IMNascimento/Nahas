"""
Helper para garantir conexões MySQL seguras em ambiente multiprocesso.
Coloque este arquivo em: src/database/db_multiprocess.py
"""
import os
from functools import wraps


def ensure_process_db_connection(func):
    """
    Decorator que garante uma conexão fresca ao banco em cada processo.
    Use em funções que rodam em workers paralelos.
    
    Exemplo:
        @ensure_process_db_connection
        def train_model(...):
            # código que acessa o banco
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        from database.model_base import db
        
        pid = os.getpid()
        
        # Fecha conexão antiga (se existir)
        if not db.is_closed():
            try:
                db.close()
            except Exception:
                pass
        
        # Cria nova conexão para este processo
        try:
            db.connect(reuse_if_open=False)
            print(f"[PID-{pid}] Nova conexão MySQL estabelecida", flush=True)
        except Exception as e:
            print(f"[PID-{pid}] ERRO ao conectar: {e}", flush=True)
            raise
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Fecha ao terminar
            try:
                if not db.is_closed():
                    db.close()
                    print(f"[PID-{pid}] Conexão MySQL fechada", flush=True)
            except Exception:
                pass
    
    return wrapper


def get_process_safe_db():
    """
    Retorna uma conexão segura ao banco para o processo atual.
    Use em contextos onde você precisa de acesso manual.
    
    Exemplo:
        db = get_process_safe_db()
        query = MyModel.select()
    """
    from database.model_base import db
    
    if db.is_closed():
        db.connect(reuse_if_open=False)
    
    return db


class ProcessSafeDBContext:
    """
    Context manager para garantir conexão segura.
    
    Exemplo:
        with ProcessSafeDBContext():
            data = HourlyQuoteBitcoin.get_between_dates(...)
    """
    def __enter__(self):
        from database.model_base import db
        
        if db.is_closed():
            db.connect(reuse_if_open=False)
        
        return db
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        from database.model_base import db
        
        try:
            if not db.is_closed():
                db.close()
        except Exception:
            pass
        
        return False  # Não suprime exceções