import datetime
import logging
import os
from logging import FileHandler, StreamHandler, Formatter

from colorlog import ColoredFormatter

from src.main.utils.Path import getResourcePath

# ____
LOG_PATH = os.path.join(getResourcePath(),"log")
# _____________
LOG_FILE_LEVEL = logging.DEBUG
LOG_CONSOLE_LEVEL = logging.INFO

FILE_LOG_FORMAT = '[%(asctime)s][%(processName)s][%(levelname)s]' \
                  '[%(filename)s:%(funcName)s:%(lineno)d]:%(message)s'
CONSOLE_LOG_FORMAT = '[%(asctime)s][%(processName)s][%(levelname)s]' \
                     '[%(filename)s:%(funcName)s:%(lineno)d]:%(log_color)s\n%(message)s'


# ________（_______，______）
def ensure_directory_exists(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except PermissionError as e:
            print(f"____，______: {path}")
            raise e


ensure_directory_exists(LOG_PATH)


class Logger:
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)
        self._init_loggers()

    def _init_loggers(self):
        self._add_file_handler(self.logger, LOG_FILE_LEVEL)
        self._add_console_handler(self.logger, LOG_CONSOLE_LEVEL)

    def _add_file_handler(self, logger, level):

        # ______，_________
        current_time = datetime.datetime.now()
        # __________
        script_name = os.path.basename(__file__)
        script_name = script_name.split('.')[0]
        log_filename = f"{script_name}_{current_time.strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = FileHandler(os.path.join(LOG_PATH, log_filename))
        file_formatter = Formatter(FILE_LOG_FORMAT, datefmt='%H:%M:%S')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    def _add_console_handler(self, logger, level):
        # ______

        console_handler = StreamHandler()
        console_formatter = ColoredFormatter(
            CONSOLE_LOG_FORMAT, datefmt='%H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'bold_red',

            }
        )
        # console_formatter = Formatter(CONSOLE_LOG_FORMAT,datefmt='%H:%M:%S')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(level)
        logger.addHandler(console_handler)


logprint = Logger().logger

if __name__ == '__main__':
    # ____
    logprint.info("Informational message")
    logprint.debug("Informational message")
    logprint.error("Error message")
