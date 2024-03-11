import logging
from datetime import datetime
import os
import sys



Log_File = f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'
log_path = os.path.join(os.getcwd(),"logs",Log_File)
os.makedirs(log_path,exist_ok=True)

Log_File_path = os.path.join(log_path,Log_File)

logging.basicConfig(
    filename = Log_File_path,
    format= "[%(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

