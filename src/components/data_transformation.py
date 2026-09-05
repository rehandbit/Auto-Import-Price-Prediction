import os, sys
import numpy as np
import pandas as pd

from src.logger import logger
from src.exception import CustomException

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.utils import save_object

class DataTransformation:
    def __init__(self):
        self.scaler_path = os.path.join('artifacts', 'preprocessor', 'scaler.pkl')
        self.ohe_path = os.path.join('artifacts', 'preprocessor', 'ohe.pkl')

    def initiate_data_transformation(self,train_path, test_path):
        try:
            logger.info('Data Transformationo Started -\/')
            #Load ---------------------------------
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info('Train and Test data loaded')

            # Clean ---------------------------------

            numerix_fix = ['normalized_losses', 'peak_rpm', 'horsepower']
            for c in numerix_fix:
                train_df[c] = pd.to_numeric(train_df[c], errors='coerce')
                train_df[c] = train_df[c].fillna(train_df[c].median())
                train_df[c] = train_df[c].astype('int64')


                test_df[c] = pd.to_numeric(test_df[c], errors='coerce')
                test_df[c] = test_df[c].fillna(train_df[c].median())
                test_df[c] = test_df[c].astype('int64')

            # num of cylinder (word to int)( two --> 2)
            word_to_int = {
                "two" : 2, "three" : 3, "four" : 4 , "five" : 5 , "six" : 6, "eight" : 8, "twelve" : 12
                }
            train_df['num_of_cylinders'] = train_df['num_of_cylinders'].map(word_to_int)
            test_df['num_of_cylinders'] = train_df['num_of_cylinders'].map(word_to_int)

            # num of door ---------------------------------
            door_map = {"two": 2, "four": 4}
            train_df["num_of_doors"] = train_df["num_of_doors"].map(door_map)
            test_df["num_of_doors"] = test_df["num_of_doors"].map(door_map)

            #drop missing price ---------------------------------
            train_df = train_df.dropna(subset=['price']).reset_index(drop=True)
            test_df = test_df.dropna(subset=['price']).reset_index(drop=True)

            logger.info('Cleaning Done -\/')

            # Feature ---------------------------------
            TARGET = 'price'

            X_train =  train_df.drop(columns = [TARGET])
            y_train = train_df[TARGET]
            
            X_test = train_df.drop(columns = [TARGET])
            y_test = test_df[TARGET]

            numeric_feature = X_train.select_dtypes(include = [np.number]).columns.tolist()
            categorical_feature = X_train.select_dtypes(exclude = [np.number]).columns.to_list()

            logger.info(f'Numerical Feature : {len(numeric_feature)}')
            logger.info(f'Categorical Feature : {len(categorical_feature)}')

            # Scaler ---------------------------------
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train[numeric_feature])
            X_test_scaled = scaler.transform(X_test[numeric_feature])

            logger.info('Scaling done')

            # OneHotEncoding ---------------------------------
            ohe = OneHotEncoder(handle_unknown= 'ignore', sparse_output=False)
            X_train_ohe = ohe.fit_transform(X_train[categorical_feature])
            X_test_ohe = ohe.transform(X_test[categorical_feature])

            logger.info('One Hot Encoding done')

            # COmbine ---------------------------------
            X_train_final = np.hstack([X_train_scaled, X_train_ohe])
            X_test_final = np.hstack([X_test_scaled, X_test_ohe])
            logger.info('COmbining done')
            logger.info(f'Final train shape : {X_train_final}')
            logger.info(f'Final test shape : {X_test_final}')

            # Saving Scaler and one hot encoding
            save_object(self.scaler_path, scaler)
            save_object(self.ohe_path,    ohe)

            logger.info('Scaler and OHE Saved -\/')

            return X_train_final, X_test_final, y_train, y_test

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    from src.components.data_ingestion import DataIngestion

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()


    transformation = DataTransformation()
    X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(train_path, test_path)

    print('X_train shape : ', X_train.shape)
    print('X_test shape : ', X_test.shape)
    print('y_train shape : ', y_train.shape)
    print('y_test shape : ', y_test.shape)
