import sys
import numpy as np
import pandas as pd

from src.logger import logger
from src.exception import CustomException
from src.utils import load_object
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class ModelEvaluation:
    def initiate_model_evaluation(self,x_test, y_test):
        try:
            logger.info('Model evaluation started')

            model = load_object('artifacts/model/model.pkl')

            y_pred  = model.predict(x_test)
            r2      = r2_score(y_test, y_pred)
            rmse    = np.sqrt(mean_squared_error(y_test, y_pred))
            mae     = mean_absolute_error(y_test, y_pred)

            result_df = pd.DataFrame({
                'Actual Price '   : y_test.values,
                'Predicted Price' : y_pred.round(0),
                'Difference'      : ((y_test.values- y_pred).round(0)),
                'Difference %'    : ((abs(y_test.values - y_pred) / y_test.values)*100).round(1)
            })

            print('\n ---Model Evaluation ---')
            print(f'R2   : {r2:.2f}')
            print(f'RMSE : {rmse:.2f}')
            print(f'MAE  : {mae:.2f}')

            print(result_df.to_string())

            within_10   = (result_df['Difference %']<=10).sum()
            total       = len(result_df)
            print(f'\n {within_10}/{total} Prediction within 10 error')
            print(f'Accuracy within 10% {within_10/total*100:.1f}%')

            logger.info(f"R2: {r2}  RMSE: {rmse}  MAE: {mae}")
            logger.info(f"Accuracy within 10%: {round(within_10/total*100, 1)}%")

            return r2, rmse, mae


        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    transformation = DataTransformation()
    x_train, x_test, y_train , y_test = transformation.initiate_data_transformation(train_path, test_path)

    evaluation =ModelEvaluation()
    evaluation.initiate_model_evaluation(x_test, y_test)