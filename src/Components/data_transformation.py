import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import sys
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
from src.utils import save_path

@dataclass
class Data_Transformation_config:
    preprocessor_path = os.path.join("Artifects",'preprocessor.pkl')

class Data_transformation:
    def __init__(self):
        self.Transformation_config = Data_Transformation_config()

    def Data_Transformer(self):
        try:
            numerical_data = ['reading score', 'writing score']
            categorical_data = ['gender',
                                'race/ethnicity',
                                'parental level of education',
                                'lunch',
                                'test preparation course']
            
            num_pipeline = Pipeline(
                steps= [
                    ('Imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder())
                ]
                    )
                    
                

            logging.info("Scaling of Numerical data completed!")

            logging.info("Categorical data is successfully encoded and scaled!")


            preprocessor = ColumnTransformer(
                [
                    ("numerical_pipeline", num_pipeline,numerical_data),
                    ('Categorical_pipeline', cat_pipeline,categorical_data)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiateDataTransformation(self,Train_data,Test_data):

        try:
            Train_df = pd.read_csv(Train_data)
            Test_df = pd.read_csv(Test_data)

            preprocesser_obj = self.Data_Transformer()

            Target_att = 'math score'
            numerical_col = ['reading score', 'writing score']

            New_Train_Data_att = Train_df.drop([Target_att],axis=1)
            New_Train_Data_target_att = Train_df[Target_att]

            New_Test_Data_att = Test_df.drop([Target_att],axis=1)
            New_Test_Data_target_att = Test_df[Target_att]

            logging.info('Applying dataTransformatin on training and test data!')
            
            training_transformation = preprocesser_obj.fit_transform(New_Train_Data_att)
            test_transformation = preprocesser_obj.transform(New_Test_Data_att)

            train_arr = np.c_[training_transformation,np.array(New_Train_Data_target_att)]
            test_arr = np.c_[test_transformation,np.array(New_Test_Data_target_att)]

            logging.info('Training and Test data has successfully Transformed!')
            save_path(
                file_path = self.Transformation_config.preprocessor_path,
                obj = preprocesser_obj

            )
            logging.info("Pickel file has been saved")
            return (
                train_arr,test_arr,self.Transformation_config.preprocessor_path
            )
        except Exception as e:
            raise CustomException(e,sys)
