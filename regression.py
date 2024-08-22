import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

models = [
    LinearRegression(),
    Ridge(alpha=1.0),
    Lasso(alpha=0.1),
    RandomForestRegressor(n_estimators=100, random_state=42),
    SVR(kernel='rbf')
]
result = pd.DataFrame(columns=["Model", "Avg r2", "Avg mse"])

def load_data(filepath, delimiter):
    data = pd.read_csv(filepath, delimiter=delimiter) 
    return data

def preprocess_data(data, target_col):
    X = data.drop(target_col, axis=1) # everything in file other than target 
    y = data[target_col]
    
    # scale data 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def perform_cross_validation(X, y, model, n_splits=5, random_state=42):
    # K-Fold cross-validation 
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    mse_scores = []
    r2_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mse_scores.append(mse)
        r2_scores.append(r2)
        
    avg_mse = np.mean(mse_scores)
    avg_r2 = np.mean(r2_scores)
    return avg_r2, avg_mse

def main():
    # get user inputs for file path and target column
    filepath = input("Enter the .csv file path: ")
    target_col = input("Enter the target column name: ")
    delimiter = input("Enter the delimiter for the file: ")
    
    # load and preprocess the data
    data = load_data(filepath, delimiter)
    X, y = preprocess_data(data, target_col)

    global result
    
    # cycle through models to run
    for model in models:
        avg_r2, avg_mse = perform_cross_validation(X, y, model)
        # append the results to the df
        result = result.append({
            "Model": model.__class__.__name__,
            "Avg r2": avg_r2,
            "Avg mse": avg_mse
        }, ignore_index=True)

    print (result)


if __name__ == '__main__':
    main()