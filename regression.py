import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

def load_data(filepath, delimiter):
    data = pd.read_csv(filepath, delimiter=delimiter)
    # print(data.columns) # verify columns 
    return data

def preprocess_data(data, target_col):
    features = data.drop(target_col, axis=1) # everything in file other than target 
    target = data[target_col]
    
    # scale data 
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, target

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
        
    return mse_scores, r2_scores

def print_results(mse_scores, r2_scores):
    print(f'Mean Squared Error for each fold: {mse_scores}')
    print(f'Average MSE across all folds: {np.mean(mse_scores)}')
    print(f'R-squared for each fold: {r2_scores}')
    print(f'Average R-squared across all folds: {np.mean(r2_scores)}')


def main():
    # get user inputs for file path and target column
    filepath = input("Enter the .csv file path: ")
    target_col = input("Enter the target column name: ")
    delimiter = input("Enter the delimiter for the file: ")
    
    # load and preprocess the data
    data = load_data(filepath, delimiter)
    X, y = preprocess_data(data, target_col)
    
    # specify the model to use
    model = LinearRegression()
    
    # perform cross-validation and print results
    mse_scores, r2_scores = perform_cross_validation(X, y, model)
    print_results(mse_scores, r2_scores)

if __name__ == '__main__':
    main()