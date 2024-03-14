from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.prediction_pipeline import DataFromUser,PredictPipeline


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/Predict",methods = ['GET', 'POST'])
def predict_data():
    if request.method == 'GET':
        return render_template('Home.html')
    else:
        data = DataFromUser(gender = request.form.get("gender"),
        race_ethnicity = request.form.get("ethnicity"),
        parental_level_of_education = request.form.get("parental_level_of_education"),
        lunch = request.form.get("lunch"),
        test_preparation_course = request.form.get("test_preparation_course"),
        reading_score = request.form.get("reading_score"),
        writing_score = request.form.get("writing_score")
        )

        pred_df = data.data_as_dataframe()
        pred_df.to_csv(r"C:\Users\49179\Desktop\Student Performance prediction\Data\datafromweb.csv",index=False,header=True)
        predictpipeline_ = PredictPipeline()
        result = predictpipeline_.prediction(pred_df)
        return render_template('Home.html',results = result[0] )
    
if __name__  == "__main__":
    app.run(host="0.0.0.0")