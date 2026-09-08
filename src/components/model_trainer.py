import os, sys
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_error

from src.logger import logger
from src.exception import CustomException
from src.utils import save_object


class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model", "model.pkl")

    def evaluate_model(self, X_train, X_test, y_train, y_test, models):
        try:
            result ={}
            

            for name, model in models.items():
                model.fit(X_train, y_train)
                
                y_test_pred = model.predict(X_test)

                r2 = r2_score(y_test, y_test_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                mae = mean_absolute_error(y_test, y_test_pred)
###
                result[name] = {
                    "model" : model,
                    "r2" : r2,
                    'rmse' :rmse,
                    'mae' :mae,
                }

                logger.info(f'{name:25s} , r2 = {r2}, rmse = {rmse}, mae = {mae:2f}')
            return result



        except Exception as e:
            raise CustomException(e,sys)


    def inititate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            logger.info('model training started -\/')
            logger.info(f'X_train NaN: {np.isnan(X_train).sum()}')
            logger.info(f'X_test NaN: {np.isnan(X_test).sum()}')


            # All Model define -----------------------

            models = {
                'Linear Regression' : LinearRegression(),
                'Ridge' : Ridge(alpha=10.0),
                'Lasso' : Lasso(alpha=10.0, max_iter=100000),
                'Decision Tree': DecisionTreeRegressor(random_state=42),
                'KNN': KNeighborsRegressor(),
                'Random Forest': RandomForestRegressor(n_estimators=300 , random_state=42),
                'ADA Boost': AdaBoostRegressor(random_state=42),
                'Cat Boost': CatBoostRegressor(verbose=False, random_state=42),
                'XGBoost': XGBRegressor(learning_rate=0.05, max_depth=3, n_estimators=300 , random_state=42),
            }
            results = self.evaluate_model(X_train, X_test, y_train, y_test, models)

            # Best model
            best_name = max(results, key=lambda x: results[x]['r2'])
            best_r2 = results[best_name]['r2']
            best_model = results[best_name]['model']

            print(f'best model : {best_model}')
            print(f'Best R2 : {best_r2}')

            logger.info(f'Best Model : {best_name} -> R2 : {best_r2}')

            save_object(self.model_path, best_model)
            logger.info(f'best Model save at : {self.model_path}')

            return best_name, best_r2
            
        except Exception as e:
            raise CustomException(e,sys)
        

if __name__ == '__main__':
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(train_path, test_path)
    
    trainer = ModelTrainer()
    best_name, best_r2 = trainer.inititate_model_trainer(X_train, X_test, y_train, y_test)
    
    print(f'\n -- Final best Model : {best_name}')
    print(f'\n -- Final best R2    : {best_r2}')
