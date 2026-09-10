import os, sys, pickle
import numpy as np
import pandas as pd
from src.logger import logger
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)

        logger.info(f'object save at: {file_path}')

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f'Object Loaded from: {file_path}')
        return obj
    
    except Exception as e:
        raise CustomException(e, sys)