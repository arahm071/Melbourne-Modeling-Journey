# Standard library imports
import os              # Provides a way of using operating system dependent functionality

# Third-party imports
import pandas as pd           # Data manipulation and analysis library
import numpy as np            # Scientific computing library
import matplotlib.pyplot as plt  # Plotting library for creating static and interactive visualizations
import seaborn as sns         # Data visualization library based on matplotlib
import statsmodels.api as sm  # Statistical models including OLS regression
import scipy.stats as stats   # Scientific computing and statistics library
from math import log                                                                  # Necessary for logarithmic operations
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score, GridSearchCV  # Model selection and evaluation tools
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression                                     # Linear models for regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error         # Metrics for model evaluation
from sklearn.pipeline import make_pipeline
from statsmodels.stats.stattools import durbin_watson                                 # Durbin-Watson statistic for detecting autocorrelation
from statsmodels.stats.diagnostic import het_breuschpagan                             # Breusch-Pagan test for heteroscedasticity
from statsmodels.tools.tools import add_constant                                      # Adds a constant term to an array for modeling

# * File Importation

# Get the absolute path to the directory of the current script
script_dir = os.path.abspath(os.path.dirname(__file__))

# Build the absolute path to the data file
data_path = os.path.join(script_dir, '..', '02_Exploratory_Data_Analysis', '2_transformed_melb_data.csv')

# Load the transformed Melbourne housing dataset
melb_data = pd.read_csv(data_path)

# * Prepare Data For Model Fitting

# Creating dummy variables for categorical columns
dummies_Suburb = pd.get_dummies(melb_data['Suburb'], drop_first=True, dtype=int)
dummies_Regionname = pd.get_dummies(melb_data['Regionname'], drop_first=True, dtype=int)
dummies_Type = pd.get_dummies(melb_data['Type'], drop_first=True, dtype=int)
dummies_Method = pd.get_dummies(melb_data['Method'], drop_first=True, dtype=int)

# Concatenating the dummy variables with the main DataFrame
melb_data = pd.concat([melb_data, dummies_Regionname, dummies_Type, dummies_Method], axis=1)

# Dropping columns not used in the regression model
excluded_columns = ['Address', 'Suburb', 'Regionname', 'Type', 'Method', 'Date', 'Propertycount']
melb_data.drop(excluded_columns, axis=1, inplace=True)

# Assign Independent and Dependent variables
X = melb_data.drop('Price_boxcox', axis=1)
y = melb_data['Price_boxcox']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# * GridSearch and Cross-Validation 

'''
# Setting up the pipeline for polynomial regression
pipeline = make_pipeline(PolynomialFeatures(), LinearRegression())

# Defining the parameter grid: test degrees 1 through 5
param_grid = {'polynomialfeatures__degree': [1, 2, 3, 4, 5]}

# Setting up the GridSearchCV object
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')

# Fitting the grid search
grid_search.fit(X_train, y_train)

# Best degree
best_degree = grid_search.best_params_['polynomialfeatures__degree']
print(f"Best polynomial degree: {best_degree}")
'''

# * Model Fitting

# Create a polynomial regression model
degree = 1
print(f'Degree Choosen For Model: {degree}')
poly_features = PolynomialFeatures(degree, include_bias=True)
poly_model = make_pipeline(poly_features, LinearRegression())

# Fitting the model
model = poly_model.fit(X_train, y_train)

# Evaluating the model
train_score = poly_model.score(X_train, y_train)
test_score = poly_model.score(X_test, y_test)

print(f"Training score: {train_score}")
print(f"Test score: {test_score}")

# Predicting on the test data
y_pred = model.predict(X_test)

# Calculating the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculating the mean absolute error
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae}")

# Calculate fitted values (predictions on the training set, a.k.a. y_training_predictions)
fitted_values = model.predict(X_train)

# Calculate residuals
residuals = y_train - fitted_values

# Calculate test residuals (prediction errors)
test_residuals = y_test - y_pred

# * Model Diagnostics

def calculate_aic_bic(n, mse, p):
    """
    Calculate Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC).

    Parameters:
    - n (int): The number of observations.
    - mse (float): Mean Squared Error of the model.
    - p (int): The number of predictors (independent variables) in the model.

    Returns:
    - tuple: A tuple containing the AIC and BIC values.
    """
    aic = n * log(mse) + 2 * p
    bic = n * log(mse) + log(n) * p
    return aic, bic

def calculate_adjusted_r_squared(r_squared, n, p):
    """
    Calculate the adjusted R-squared value for a regression model.

    Adjusted R-squared accounts for the number of predictors in the model, providing a
    metric that penalizes for excessive use of uninformative predictors.

    Parameters:
    - r_squared (float): The R-squared value from the model.
    - n (int): The total number of observations in the dataset.
    - p (int): The number of predictors (independent variables) in the model.

    Returns:
    - float: Adjusted R-squared value.
    """
    adjusted_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
    return adjusted_r_squared

# Residual Analysis
plt.scatter(fitted_values, residuals)
plt.title('Poly, Residuals vs Predicted')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

# Diagnostic Plots (Q-Q Plot)
stats.probplot(residuals, dist="norm", plot=plt)
plt.show()

#Condition Number
X_poly = poly_features.fit_transform(X_train)
condition_number = np.linalg.cond(X_poly)
print("Condition Number:", condition_number)

# Checking for Autocorrelation (Durbin-Watson Test)
dw = durbin_watson(residuals)
print("Durbin-Watson statistic:", dw)

# Scale-Location Plot (also known as Spread-Location Plot)
plt.figure()
plt.scatter(fitted_values, np.sqrt(np.abs(residuals)))
plt.xlabel('Fitted Values')
plt.ylabel('Sqrt(Abs(Residuals))')
plt.title('Scale-Location Plot')
plt.show()

# Performing Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validated scores:", scores)

# R-squared
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# Adjusted R-squared
n =  len(y_test)
p = X_test.shape[1]
adj_r2 = calculate_adjusted_r_squared(r2, n, p)
print(f"Adjusted R-squared: {adj_r2}")

#AIC and BIC
aic, bic = calculate_aic_bic(n, mse, p)
print(f"AIC: {aic}, BIC: {bic}")

# Jarque-Bera Test
jb_statistic, jb_p_value = stats.jarque_bera(residuals)
print(f"Jarque-Bera Test: Statistic: {jb_statistic}, p-value: {jb_p_value}")

# Breusch-Pagan Test
bp_test = het_breuschpagan(residuals, add_constant(X_train))
print(f"Breusch-Pagan Test: {bp_test}")
