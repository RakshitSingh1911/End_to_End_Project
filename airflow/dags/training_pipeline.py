from __future__ import annotations
import json
from textwrap import dedent
import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator
from Insurance.entity import config_entity
from Insurance.components.data_ingestion import DataIngestion
from Insurance.components.data_validation import DataValidation
from Insurance.components.data_transformation import DataTransformation
from Insurance.components.model_trainer import ModelTrainer
from Insurance.components.model_evaluation import ModelEvaluation


training_pipeline_config = config_entity.TrainingPipelineConfig()

with DAG(
    "Insurance_training_pipeline",
    default_args={"retries": 2},
    description="it is my training pipeline",
    schedule="@weekly",# here you can test based on hour or mints but make sure here you container is up and running
    start_date=pendulum.datetime(2024, 1, 30, tz="UTC"),
    catchup=False,
    tags=["machine_learning ","classification","Insurance"],
) as dag:
    
    dag.doc_md = __doc__
    
    def data_ingestion(**kwargs):
        ti = kwargs["ti"]
        data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        ti.xcom_push("data_ingestion_artifact", {"train_data_path":data_ingestion.train_file_path,"test_data_path":data_ingestion.test_data_path})

    def data_transformations(**kwargs):
        ti = kwargs["ti"]
        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config, 
        data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        train_arr= data_transformation.transformed_train_path
        test_arr= data_transformation.transformed_test_path
        ti.xcom_push("data_transformations_artifcat", {"train_arr":train_arr,"test_arr":test_arr})

    def model_trainer(**kwargs):
        import numpy as np
        ti = kwargs["ti"]
        data_transformation_artifact = ti.xcom_pull(task_ids="data_transformation", key="data_transformations_artifcat")
        model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact=data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

    
    ## you have to config azure blob
    def push_data_to_azureblob(**kwargs):
        import os
        bucket_name="reposatiory_name"
        artifact_folder="/app/artifacts"
        #you can save it ti the azure blob
        #os.system(f"aws s3 sync {artifact_folder} s3:/{bucket_name}/artifact")
        
        
        
    data_ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=data_ingestion,
    )
    data_ingestion_task.doc_md = dedent(
        """\
    #### Ingestion task
    this task creates a train and test file.
    """
    )

    data_transform_task = PythonOperator(
        task_id="data_transformation",
        python_callable=data_transformations,
    )
    data_transform_task.doc_md = dedent(
        """\
    #### Transformation task
    this task performs the transformation
    """
    )

    model_trainer_task = PythonOperator(
        task_id="model_trainer",
        python_callable=model_trainer,
    )
    model_trainer_task.doc_md = dedent(
        """\
    #### model trainer task
    this task perform training
    """
    )
    



data_ingestion_task >> data_transform_task >> model_trainer_task 