import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib

if __name__ == '__main__':
    # Load UCI Heart Disease (Cleveland) dataset
    heart = fetch_openml('heart-disease', version=1, as_frame=True)
    X = heart.data

    # Convert all features to numeric (some may be categorical strings)
    X = X.apply(pd.to_numeric, errors='coerce')

    # In some sklearn versions the target ends up inside heart.data
    if 'target' in X.columns:
        y = X.pop('target')
    else:
        y = pd.to_numeric(pd.Series(heart.target), errors='coerce')

    # Binarize target: 0 = no disease, 1 = disease (original has 0-4)
    y = y.fillna(0)
    y = (y > 0).astype(int)

    # Fill missing values with column medians
    X = X.fillna(X.median())

    feature_names = list(X.columns)
    print(f"Features: {feature_names}")

    # 80/20 train-test split with fixed seed for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Pipeline: standardize features -> Random Forest
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"Test accuracy: {accuracy:.3f}")

    joblib.dump(model, 'model.joblib')
    joblib.dump(feature_names, 'feature_names.joblib')
    print("Model and feature names saved.")
