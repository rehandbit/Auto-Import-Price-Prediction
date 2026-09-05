import os, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logger
from src.exception import CustomException

class DataIngestion:
    def __init__(self):
        self.raw_data_path   = os.path.join('data', 'raw', 'auto_imports.csv')
        self.train_data_path = os.path.join('data', 'processed', 'train.csv')
        self.test_data_path  = os.path.join('data', 'processed', 'test.csv')

    def initiate_data_ingestion(self):
        try:
            
            logger.info('Data Ingestion started -\/')

            #adding column since this data does not have header
            columns = [
                'symboling', 'normalized_losses', 'make', 'fuel_type', 'aspiration', 'num_of_doors', 'body_style', 'drive_wheels',
                'engine_location', 'wheel_base', 'length', 'width', 'height', 'curb_weight', 'engine_type', 'num_of_cylinders',
                'engine_size', 'fuel_system', 'bore', 'stroke', 'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg', 
                'highway_mpg', 'price'
            ]

            #load
            df = pd.read_csv(self.raw_data_path, header = None, names = columns, na_values = ['?'])
            logger.info(f'data loaded - shape : {df.shape}')

            #train test split
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

            #save train_df, test_df
            os.makedirs(os.path.dirname(self.train_data_path), exist_ok=True)
            train_df.to_csv(self.train_data_path, index = False)
            test_df.to_csv(self.test_data_path, index = False)

            logger.info('Train and Test split done')
            logger.info(f'Train shape : {train_df.shape}   Test shape : {test_df.shape} ')

            return self.train_data_path, self.test_data_path


        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_path, test_path = obj.initiate_data_ingestion()
    print("Train :- ", train_path)
    print('test :- ', test_path)