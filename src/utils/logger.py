import logging
import os

class Logger:
    """
    Classe de logger profissional com métodos estáticos.
    Gerencia logs para console e arquivos de forma centralizada.
    """
    LOG_FILE = None
    LOGGER = None

    @staticmethod
    def initialize(log_file: str = None, level=logging.INFO):
        """
        Inicializa o logger com um arquivo de log opcional e configura o nível de log.
        
        :param log_file: Caminho do arquivo de log. Se None, os logs serão apenas exibidos no console.
        :param level: Nível de log (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        """
        # Criar um logger único, evitando a duplicação de handlers
        if Logger.LOGGER is None:
            Logger.LOGGER = logging.getLogger("GlobalLogger")
            Logger.LOGGER.setLevel(level)

            # Formato dos logs
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # Configurar log para console
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            Logger.LOGGER.addHandler(console_handler)

            # Configurar log para arquivo (se especificado)
            if log_file:
                Logger.LOG_FILE = log_file
                if not os.path.exists(os.path.dirname(log_file)):
                    os.makedirs(os.path.dirname(log_file))  # Cria o diretório se não existir
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                Logger.LOGGER.addHandler(file_handler)

    @staticmethod
    def debug(message: str):
        """Loga uma mensagem de debug."""
        if Logger.LOGGER:
            Logger.LOGGER.debug(message)

    @staticmethod
    def info(message: str):
        """Loga uma mensagem de informação."""
        if Logger.LOGGER:
            Logger.LOGGER.info(message)

    @staticmethod
    def warning(message: str):
        """Loga uma mensagem de aviso."""
        if Logger.LOGGER:
            Logger.LOGGER.warning(message)

    @staticmethod
    def error(message: str):
        """Loga uma mensagem de erro."""
        if Logger.LOGGER:
            Logger.LOGGER.error(message)

    @staticmethod
    def critical(message: str):
        """Loga uma mensagem crítica."""
        if Logger.LOGGER:
            Logger.LOGGER.critical(message)