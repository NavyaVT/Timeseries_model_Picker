import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.base import clone
import joblib
import os

def train_random_forest_model(X, y, output_path='D:\\DataGenie_DS\\models\\random_forest_model.pkl'):
    label_enc = LabelEncoder()
    y_encoded = label_enc.fit_transform(y)

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced'],
    }

    model = RandomForestClassifier(random_state=6)

    # grid search CV for hyperparameter tuning
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='average_precision', n_jobs=-1)
    grid_search.fit(X, y_encoded)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump((best_model, label_enc), output_path)
    return best_model, label_enc, best_params, best_score

def stratified_shufflesplit_cv(X, y, model, label_enc, n_splits=5, test_size=0.2):
    stratified_split = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=42)
    accuracies = []
    best_model_cv = None
    best_cv_accuracy = 0.0

    for train_index, test_index in stratified_split.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Clone is used for retraining
        fold_model = clone(model)
        fold_model.fit(X_train, label_enc.transform(y_train))
        
        y_pred = fold_model.predict(X_test)
        accuracy = accuracy_score(label_enc.transform(y_test), y_pred)
        accuracies.append(accuracy)

        if accuracy > best_cv_accuracy:
            best_cv_accuracy = accuracy
            best_model_cv = fold_model

    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)

    return accuracies, best_model_cv, label_enc, mean_accuracy, std_accuracy, best_cv_accuracy

if __name__ == "__main__":
    data = pd.read_csv("data_with_labels.csv", header=0)

    features = ['seasonality_strength', 'trend', 'period', 'adf_pvalue', 'acf1',
                'std_deviation', 'skew', 'kurtosis', 'entropy']
    X = data[features].values
    labels = data['model']

    model, label_enc, best_params, best_score = train_random_forest_model(X, labels)

    cv_scores, best_model_cv, label_enc_cv, mean_accuracy, std_accuracy, best_cv_accuracy = stratified_shufflesplit_cv(X, labels, model, label_enc)

    joblib.dump((best_model_cv, label_enc_cv), 'models\\best_model_cv.pkl')
    print("Best model and label encoder saved as 'best_model_cv.pkl'")