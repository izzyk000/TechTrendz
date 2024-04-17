import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
from preprocessing import load_data, get_preprocessor


def train_and_evaluate_model(filepath):
    df = load_data(filepath)
    if df is None:
        print("Data loading failed, stopping execution.")
        return
    
    #Model preparation
    df['High Value Purchase'] = (df['Annual Spend'] >= df['Annual Spend'].quantile(0.75)).astype(int)
    X = df.drop(['Customer ID', 'High Value Purchase', 'Last Purchase Date'], axis=1)
    y = df['High Value Purchase']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Get preprocessing and setup the pipeline
    preprocessor = get_preprocessor()
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Define the parameter grid
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_features': ['auto', 'sqrt', 'log2'],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)

    # Train the model
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Evaluation and reporting
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))
    y_pred_best = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_best))

    # Save the best model
    joblib.dump(best_model, 'best_model.pkl')
    print("Model trained and saved successfully.")

    # Validate saved model
    try:
        loaded_model = joblib.load('best_model.pkl')
        test_pred = loaded_model.predict(X_test)
        print("Loaded model validation successful, Accuracy:", accuracy_score(y_test, test_pred))
    except Exception as e:
        print(f"Failed to load or validate the model: {e}")



if __name__ == "__main__":
    train_and_evaluate_model('data/Updated_Customer_Data_Australia.xlsx')



