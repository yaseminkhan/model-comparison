# Model Comparison Tool

## Description

This project develops a Python-based tool to compare various regression models to predict a target variable from given datasets. The tool evaluates models based on their R² (R-squared) and MSE (Mean Squared Error) metrics to identify which model best fits the data.

## Features

- **Model Evaluation**: Compares Linear Regression, Ridge, Lasso, Random Forest, and Support Vector Regression models.
- **Metrics Comparison**: Evaluates models based on R² and MSE.
- **Data Preprocessing**: Includes scaling features to normalize data.
- **Cross-Validation**: Utilizes K-Fold cross-validation to ensure model reliability.

## Models Used

- `LinearRegression`: Simple linear regression model for predicting quantitative outcomes.
- `Ridge`: Linear regression with L2 regularization.
- `Lasso`: Linear regression with L1 regularization, useful for feature selection.
- `RandomForestRegressor`: An ensemble of decision trees for regression tasks, robust to overfitting.
- `SVR`: Epsilon-Support Vector Regression.

## Prerequisites

Before you run this tool, ensure you have the following installed:
- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/yaseminkhan/model-comparison.git
cd model-comparison
```

## Usage

To run this tool, navigate to the cloned directory, and execute:

```bash
python model_comparison.py
```

You will be prompted to enter:
- The path to your CSV data file.
- The name of the target column in your dataset.
- The delimiter used in your CSV file.

## Example

Input:

```plaintext
Enter the .csv file path: /path/to/data.csv
Enter the target column name: Price
Enter the delimiter for the file: ,
```

Output:

```plaintext
Model                Avg r2     Avg mse
0       LinearRegression  0.274890  0.568145
1                  Ridge  0.274986  0.568069
2                  Lasso  0.211574  0.617890
3  RandomForestRegressor  0.532308  0.366801
4                    SVR  0.394660  0.473874
```

## Model Specifications

### Data Requirements
- **Data Type**: This model requires all input features to be numeric. Users must convert categorical data to numeric formats using encoding strategies before using this tool.
- **Preprocessing**: Data should be scaled or normalized to ensure accurate results. The model uses `StandardScaler` from Scikit-Learn to scale features.

### Operational Limitations
- **Feature Independence**: The model evaluates each feature's impact on the target variable independently, not accounting for potential interactions between features.
- **Feature Configuration**: Users cannot specify subsets of features directly through the model's interface. Instead, adjust your dataset prior to input to include only the relevant features.

### Use Case Guidelines
- **Optimal Use**: The model is best suited for quantitative datasets with clear numeric relationships and minimal missing values.
- **Limitations**: Performance may degrade with high-dimensional data or datasets where feature interactions significantly impact the target variable.

Please ensure your data conforms to the above specifications for optimal model performance.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.