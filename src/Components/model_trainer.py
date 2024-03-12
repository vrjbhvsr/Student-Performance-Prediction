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
from dataclasses import dataclass
from src.logger import logging


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

        except:
            pass