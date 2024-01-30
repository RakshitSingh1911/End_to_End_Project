import os
import sys
import pandas as pd
from Insurance.entity.config_entity import TRANSFORMER_OBJECT_FILE_NAME,MODEL_FILE_NAME,TARGET_ENCODER_OBJECT_FILE_NAME
from Insurance.logger import logging
from Insurance.exception import InsuranceException
from Insurance.utils import load_object

from Insurance.entity import config_entity
PIPELINE_PATH = config_entity.TrainingPipelineConfig()
TRANSFORMER_OBJECT_PATH = config_entity.DataTransformationConfig(PIPELINE_PATH).transform_object_path
ENCODER_DECODER_PATH  = config_entity.DataTransformationConfig(PIPELINE_PATH).target_encoder_path
MODEL_PATH = config_entity.ModelTrainerConfig(PIPELINE_PATH).model_path

class PredictPipeline:
    def __init__(self):
        pass
    
    def PREDICT(self,INPUT_FEATURES):
        try:
            TRANSFORMER_OBJECT = load_object(TRANSFORMER_OBJECT_PATH)
            ENCODER_DECODER_OBJECT =  load_object(ENCODER_DECODER_PATH)
            MODEL_OBJECT =load_object(MODEL_PATH)

            for FEATURE in INPUT_FEATURES.columns:
                if INPUT_FEATURES[FEATURE].dtypes == 'O':
                    INPUT_FEATURES[FEATURE] = ENCODER_DECODER_OBJECT.fit_transform(INPUT_FEATURES[FEATURE])
                    
            
            TRANSFORMER_DATA_ARR = TRANSFORMER_OBJECT.transform(INPUT_FEATURES)
            
            pred= MODEL_OBJECT.predict(TRANSFORMER_DATA_ARR)
            
            return pred           
        
        except Exception as e:
            raise InsuranceException(e,sys)
        
    
    
class CustomData:
    def __init__(self,
                 age:float,
                 sex:str,
                 bmi:float,
                 children:float,
                 smoker:str,
                 region:str,
                 ):
        
        self.age=age
        self.children=children
        self.smoker=smoker
        self.region = region
        self.bmi = bmi
        self.sex = sex
            
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'age':[self.age] ,
                    'sex' : [self.sex],
                    'bmi' : [self.bmi],
                    'children' : [self.children],
                    'smoker' : [self.smoker],
                    'region' : [self.region],
                    
                    
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                raise InsuranceException(e,sys)
            

