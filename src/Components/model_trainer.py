import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import  AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostClassifier
from sklearn.svm import SVR
from xgboost import XGBRegressor
from src.utils import save_path
from src.utils import predict_model
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import os
import sys

@dataclass
class Model_Training_Config:
   train_model_file_path = os.path.join("Artifects", 'model.pkl')

class Model_Trainer:
    def __init__(self):
        self.Model_trainer_config = Model_Training_Config()

    def InitiateModelTraining(self,train_array, test_array):
        try:
            logging.info("Train and test data split")

            X_train,X_test,Y_train,Y_test = (train_array[:,:-1],test_array[:,:-1],train_array[:,-1],test_array[:,-1])

            models = {
                    "Linear Regression": LinearRegression(),
                    "Lasso": Lasso(),
                    "Ridge": Ridge(),
                    "K-Neighbors Regressor": KNeighborsRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "AdaBoost Regressor": AdaBoostRegressor(),
                    "XGBRegressor": XGBRegressor()
                    }
            

            model_report:dict = predict_model(X_train=X_train,Y_train = Y_train, X_test = X_test, Y_test=Y_test,models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score[0] < 0.6:
                raise CustomException("Best model not found")
            
            logging.info('Best model selected')


            save_path(file_path= self.Model_trainer_config.train_model_file_path,
                obj= best_model)
            
            with open(r"C:\Users\49179\Desktop\Student Performance prediction\additional_info.txt","w") as f:
                for model, values in model_report.items():
                    f.write(f"{model}: {', '.join(map(str, values))}\n")
                f.close()
        except Exception as e:
            raise CustomException(e,sys)
            