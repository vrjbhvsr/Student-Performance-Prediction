import os
import sys
from src.logger  import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from src.Components.data_transformation import Data_transformation
from src.Components.data_transformation import Data_Transformation_config
from dataclasses import dataclass
from src.Components.model_trainer import Model_Trainer

@dataclass
class Data_ingestion_Config:
    train_data_path = os.path.join('Artifects','Train_data.csv')
    test_data_path = os.path.join('Artifects','Test_data.csv')
    raw_data_path = os.path.join('Artifects','Raw_data.csv')

class Data_ingetsion:
    def __init__(self):
        self.ingestion_config = Data_ingestion_Config()

    def data_ingestion(self):
        logging.info("Data INgestion started")
        try:
            data = pd.read_csv(r'C:\Users\49179\Desktop\Student Performance prediction\Data\StudentsPerformance.csv')
            logging.info('Data Imported Successfully!')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False, header = True)

            logging.info("Train test split started!")

            train_data, test_data = train_test_split(data,test_size=0.2,random_state = 42)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False, header = True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False, header = True)

            logging.info("Data Ingestion completed!")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        


if __name__ == "__main__":
    DI = Data_ingetsion()
    Train_data,Test_data=DI.data_ingestion()
    DT = Data_transformation()
    train_arr,test_arr,_= DT.initiateDataTransformation(Train_data,Test_data)
    modeltrainer = Model_Trainer()
    modeltrainer.InitiateModelTraining(train_array=train_arr,test_array=test_arr)


