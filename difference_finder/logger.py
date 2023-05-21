import logging
from rich.logging import RichHandler


class Logger:
    def __init__(self):
        self.name = "Finder"

    def initLogger(self):
        __logger = logging.getLogger(self.name)

        FORMAT = f"[{self.name}] >> %(message)s"
        handler = RichHandler()
        handler.setFormatter(logging.Formatter(FORMAT))

        __logger.addHandler(handler)

        __logger.setLevel(logging.INFO)
    
        return __logger