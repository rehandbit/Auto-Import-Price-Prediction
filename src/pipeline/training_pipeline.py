import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logger
from src.exception import CustomException


class TrainingPipeline:
    def start_training(self):
        try:
            logger.info('Training Pipeline started -\/')
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()

            trainsformation = DataTransformation()
            X_train, X_test, y_train, y_test = trainsformation.initiate_data_transformation(train_path, test_path)


            trainer = ModelTrainer()
            best_name, best_r2 = trainer.inititate_model_trainer(X_train, X_test, y_train, y_test)

            logger.info(f'Training Complete - Best: {best_name} and R2: {best_r2}')

            return best_name, best_r2

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    pipeline = TrainingPipeline()
    best_name, best_r2 = pipeline.start_training()
    print(f'Best Model : {best_name} and R2: {best_r2}')