import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import numpy as np



class EngagementScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, weight_pages=0.3, weight_visits=0.3, weight_frequency=0.4):
        self.weight_pages = weight_pages
        self.weight_visits = weight_visits
        self.weight_frequency = weight_frequency

    def fit(self, X, y=None):
        return self  # Nothing to do here
    
    def transform(self, X):
        if isinstance(X, np.ndarray):
            # Assuming the order of features is known and consistent
            columns = ['Age', 'Annual Spend', 'Pages Viewed Per Visit', 'Number of Visits', 'Purchase Frequency']
            X = pd.DataFrame(X, columns=columns)
        
        X = X.copy()
        X['Engagement Score'] = (self.weight_pages * X['Pages Viewed Per Visit'] +
                                self.weight_visits * X['Number of Visits'] +
                                self.weight_frequency * X['Purchase Frequency'])
        return X

    
    

def load_data(filepath):
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
        # Check if all expected columns are present
        expected_columns = ['Gender', 'Location', 'Age', 'Annual Spend', 'Purchase Frequency', 'Number of Visits', 'Pages Viewed Per Visit']
        if not all(column in df.columns for column in expected_columns):
            missing = list(set(expected_columns) - set(df.columns))
            raise ValueError(f"Missing columns in the dataset: {missing}")
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None
    return df



def get_preprocessor():
    numerical_features = ['Age', 'Annual Spend', 'Pages Viewed Per Visit', 'Number of Visits', 'Purchase Frequency']
    categorical_features = ['Gender', 'Location']

    # Define the preprocessing pipeline for numerical features including engagement score calculation
    numerical_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('engagement_score', EngagementScoreTransformer())
    ])

    # Full preprocessing for both numerical and categorical data
    preprocessor = ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', OneHotEncoder(), categorical_features),
    ], remainder='passthrough')

    return preprocessor
