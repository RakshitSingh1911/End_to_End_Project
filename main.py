from Insurance.logger import logging
from Insurance.exception import InsuranceException
from Insurance.utils import get_collection_as_dataframe
import sys, os
from Insurance.entity.config_entity import DataIngestionConfig
from Insurance.entity import config_entity
from Insurance.components.data_ingestion import DataIngestion
from Insurance.components.data_validation import DataValidation
from Insurance.components.data_transformation import DataTransformation
from Insurance.components.model_trainer import ModelTrainer
from Insurance.components.model_evaluation import ModelEvaluation
from Insurance.pipeline.single_prediction_pipeline import CustomData,PredictPipeline





#def test_logger_and_expection():
   # try:
       # logging.info("Starting the test_logger_and_exception")
        #result = 3/0
       # print(result)
       # logging.info("Stoping the test_logger_and_exception")
    #except Exception as e:
      #  logging.debug(str(e))
       # raise InsuranceException(e, sys)

if __name__=="__main__":
     try:
          #start_training_pipeline()
          #test_logger_and_expection()
       # get_collection_as_dataframe(database_name ="INSURANCE", collection_name = 'INSURANCE_PRO
         data= CustomData(

            age = 19,
            sex = 'male',
            bmi = 27.9,
            children = 5,
            smoker = 'yes',
            region = 'southwest',
        )
        # this is my final data
         final_data=data.get_data_as_dataframe()
        
         predict_pipeline=PredictPipeline()
        
         pred=predict_pipeline.PREDICT(final_data)
        
         result=round(pred[0],2)

         print(result)
          


     except Exception as e:
          print(e)