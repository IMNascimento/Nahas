from database.model_base import db  # origem do objeto; antes vinha via model_nahas

def ensure_db_connection():
    """
    Garante que a conexão com o banco esteja ativa, mesmo após longos períodos ociosos.
    Se estiver fechada ou com falha, tenta reconectar com segurança.
    """
    try:
        if db.is_closed():
            db.connect(reuse_if_open=True)
        else:
            # Tenta pingar para forçar verificação
            db.connection().ping(reconnect=True)
    except Exception as e:
        print("[WARN] Reconectando ao banco devido a erro:", e)
        try:
            db.close()
        except Exception:
            pass
        db.connect(reuse_if_open=True)