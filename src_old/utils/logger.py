import logging
import os

class Logger:
    """
    Classe de logger profissional que gerencia logs para console e arquivos de forma centralizada.
    Cada instância pode ter seu próprio arquivo de log.
    """

    def __init__(self, log_file: str = None, level=logging.INFO):
        """
        Inicializa o logger com um arquivo de log opcional e configura o nível de log.
        
        :param log_file: Caminho do arquivo de log. Se None, os logs serão exibidos apenas no console.
        :param level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        self.logger = logging.getLogger(log_file if log_file else "GlobalLogger")
        self.logger.setLevel(level)

        # Formato dos logs
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Configurar log para console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Configurar log para arquivo (se especificado)
        if log_file:
            try:
                if not os.path.exists(os.path.dirname(log_file)):
                    os.makedirs(os.path.dirname(log_file))  # Cria o diretório se não existir
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
            except Exception as e:
                print(f"Erro ao criar arquivo de log: {e}")
                raise

    def debug(self, message: str):
        """Loga uma mensagem de debug."""
        self.logger.debug(message)

    def info(self, message: str):
        """Loga uma mensagem de informação."""
        self.logger.info(message)

    def warning(self, message: str):
        """Loga uma mensagem de aviso."""
        self.logger.warning(message)

    def error(self, message: str):
        """Loga uma mensagem de erro."""
        self.logger.error(message)

    def critical(self, message: str):
        """Loga uma mensagem crítica."""
        self.logger.critical(message)

    def close(self):
        """Fecha todos os handlers de arquivo do logger."""
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)