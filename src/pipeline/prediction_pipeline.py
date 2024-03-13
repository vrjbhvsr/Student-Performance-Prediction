import pandas as pd
import sys
from src.exception import CustomException
from src.utils import load_file

import pandas as pd
import sys
from src.exception import CustomException
from src.utils import load_file

class PredictPipeline:
    def __init__(self):
        pass

    def prediction(self, features):
        try:
            model_path = 'Artifects/model.pkl'
            preprocessor_path = 'Artifects/preprocessor.pkl'
            model = load_file(file_path=model_path)
            preprocessor = load_file(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            
            predict = model.predict(data_scaled)
            return predict
        except FileNotFoundError as e:
            raise CustomException(f"File not found: {e}", sys)

class DataFromUser:
    def __init__(self, gender: str, race_ethnicity: str, parental_level_of_education: str, lunch: str, test_preparation_course: str, reading_score: int, writing_score: int):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def data_as_dataframe(self):
        try:
            customdata_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score]
            }
            return pd.DataFrame(customdata_dict)
        except Exception as e:
            raise CustomException(e, sys)


