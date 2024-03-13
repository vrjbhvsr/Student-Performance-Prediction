import os 
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.exception import CustomException


def save_path(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    

def predict_model(models,X_train,X_test,Y_train,Y_test):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            model.fit(X_train,Y_train)
            score = model.score(X_train, Y_train)
            test_prediction = model.predict(X_test)
            test_model_score= float(format(r2_score(Y_test,test_prediction),'.3f'))
            MAE_= float(format(mean_absolute_error(Y_test,test_prediction),'.3f'))
            MSE_= float(format(mean_squared_error(Y_test,test_prediction),'.3f'))

            report[list(models.keys())[i]] = test_model_score,MAE_,MSE_

        return report
    
    except Exception as e:
        raise CustomException(e,sys)