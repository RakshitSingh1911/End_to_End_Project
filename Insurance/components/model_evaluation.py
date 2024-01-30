from Insurance.predictor import ModelResolver
from Insurance.entity import config_entity,artifact_entity
from Insurance.exception import InsuranceException
from Insurance.logger import logging
from Insurance.utils import load_object
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from Insurance import utils
import pandas  as pd
import sys,os
from Insurance.config import TARGET_COLUMN
import numpy as np 
import dagshub 
import mlflow

class ModelEvaluation:

    def __init__(self,
        data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
        data_transformation_artifact:artifact_entity.DataTransformationArtifact,
        model_trainer_artifact:artifact_entity.ModelTrainerArtifact      
        ):
        try:
            logging.info(f"{'>>'*20}  Model Evaluation {'<<'*20}")

            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_transformation_artifact=data_transformation_artifact
            self.model_trainer_artifact=model_trainer_artifact

        except Exception as e:
            raise InsuranceException(e,sys)
    
    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))# here is RMSE
        mae = mean_absolute_error(actual, pred)# here is MAE
        r2 = r2_score(actual, pred)# here is r3 value
        return rmse, mae, r2



    def initiate_model_evaluation(self):
        try:
            #if saved model folder has model the we will compare 
            #which model is best trained or the model from saved model folder

            #Currently trained model objects
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model  = load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)
            

            # take tyest data for testing test data 

            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)
            y_true = test_arr[:,-1]
            # target_encoder.transform(target_df)
            # accuracy using previous trained model
            
            dagshub.init(repo_owner='RakshitSingh1911', repo_name='End_to_End_Project', mlflow=True)
            with mlflow.start_run():
                y_pred =current_model.predict(test_arr[:,:-1])

                (rmse, mae, r2) = self.eval_metrics(y_true,y_pred)

                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                mlflow.sklearn.log_model(
        sk_model=current_model,
        artifact_path="sklearn-model",
        registered_model_name="Reggression_Model",
                                )

            
        except Exception as e:
            raise InsuranceException(e,sys)


