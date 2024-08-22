import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import os

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

def print_plot(result):
    # Create a figure and a set of subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 row, 2 columns

    # Plotting Average R² values
    result.plot(kind='bar', x='Model', y='Avg r2', ax=ax1, color='green', legend=False)
    ax1.set_title('Average R² Values')
    ax1.set_ylabel('Avg R²')
    ax1.set_xticklabels(result['Model'], rotation=45)

    # Plotting Average MSE values
    result.plot(kind='bar', x='Model', y='Avg mse', ax=ax2, color='orange', legend=False)
    ax2.set_title('Average MSE Values')
    ax2.set_ylabel('Avg MSE')
    ax2.set_xticklabels(result['Model'], rotation=45)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    plt.show()

def main():
    # get user inputs for file path and target column
    filepath = input("Enter the .csv file path: ")
    target_col = input("Enter the target column name: ")
    delimiter = input("Enter the delimiter for the file: ")
    
    # attempt to load and preprocess the data
    try:
        # check if the file exists
        if not os.path.isfile(filepath):
            raise FileNotFoundError("The entered file was not found.")
        
        # check if the file is a CSV file
        if not filepath.endswith('.csv'):
            raise ValueError("The file provided is not a CSV file.")
        
        # load data
        data = pd.read_csv(filepath, delimiter=delimiter)
        
        # check if the target column is in the df
        if target_col not in data.columns:
            raise ValueError("The entered target column does not exist in the data.")
        
        # preprocess data
        X, y = preprocess_data(data, target_col)
    
    except FileNotFoundError as e:
        print(e)
        print("Please check the file path and try again.")
        return
    except ValueError as e:
        print(e)
        print("Please check the target column name and try again.")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please try again.")
        return

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

    print_plot(result)


if __name__ == '__main__':
    main()