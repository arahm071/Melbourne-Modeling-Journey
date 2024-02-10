# Standard library imports
import os              # Provides a way of using operating system dependent functionality

# Third-party imports
import pandas as pd           # Data manipulation and analysis library
import numpy as np            # Scientific computing library
import matplotlib.pyplot as plt  # Plotting library for creating static and interactive visualizations
import seaborn as sns         # Data visualization library based on matplotlib
import statsmodels.api as sm  # Statistical models including OLS regression
import scipy.stats as stats   # Scientific computing and statistics library
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score  # Model selection and evaluation tools
from sklearn.linear_model import ElasticNet, ElasticNetCV, enet_path                  # Linear models for regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error         # Metrics for model evaluation
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

# Dropping columns not used in the regression model
excluded_columns = ['Address', 'Suburb', 'Regionname', 'Type', 'Method', 'Date', 'Propertycount']
melb_data.drop(excluded_columns, axis=1, inplace=True)

# Assign Independent and Dependent variables
X = melb_data.drop('Price_log', axis=1)
y = melb_data['Price_log']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# * GridSearch and Cross-Validation 

'''
# Define a cross-validation strategy
cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)  # Reduced for testing

# Define the ElasticNetCV object with added verbosity
elastic_net_cv = ElasticNetCV(alphas=np.arange(0.00001, 0.0001, 0.0001),  # Simplified for testing
                              l1_ratio=np.arange(0.05, 1, 0.05),  # Simplified for testing
                              cv=cv, n_jobs=-1, max_iter=1000, tol=0.001, verbose=1)  # Added verbosity

# Fit the model
elastic_net_cv.fit(X_train, y_train)

# Best alpha value and l1_ratio
print(f"Optimal alpha: {elastic_net_cv.alpha_}")
print(f"Optimal l1_ratio: {elastic_net_cv.l1_ratio_}")
'''

# * Model Fitting

# Initialize the Elastic Net regression model with an alpha value
elastic_net = ElasticNet(alpha=0.00001, l1_ratio=0.9500000000000001)

# Fit the model
model = elastic_net.fit(X_train, y_train)

# Evaluate the model
train_score = elastic_net.score(X_train, y_train)
test_score = elastic_net.score(X_test, y_test)
coeff_used = np.sum(elastic_net.coef_ != 0)

print(f"Training score: {train_score}")
print(f"Test score: {test_score}")
print(f"Number of features used: {coeff_used}")

# Predict on the test data
y_pred = model.predict(X_test)

# Coefficients 
coefficients = model.coef_
print(f"Coefficients: {coefficients}")

# Calculate the mean squared error
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
plt.title('Elastic Net, Residuals vs Predicted')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

# Displaying Feature Importance
feature_importance = pd.DataFrame(model.coef_, index=X.columns, columns=['importance'])
print(feature_importance.sort_values(by='importance', ascending=False).to_string())

# Performing Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validated scores:", scores)

# Diagnostic Plots (Q-Q Plot)
stats.probplot(residuals, dist="norm", plot=plt)
plt.show()

# Checking the Condition Number
coef_matrix = np.array([model.coef_])
condition_number = np.linalg.cond(coef_matrix)
print("Condition Number:", condition_number)

# Checking for Autocorrelation (Durbin-Watson Test)
dw = durbin_watson(residuals)
print("Durbin-Watson statistic:", dw)

# Scale-Location Plot (also known as Spread-Location Plot)
fitted_values = model.predict(X_train)
plt.figure()
plt.scatter(fitted_values, np.sqrt(np.abs(residuals)))
plt.xlabel('Fitted Values')
plt.ylabel('Sqrt(Abs(Residuals))')
plt.title('Scale-Location Plot')
plt.show()

# Compute the Elastic Net path with enet_path function
alphas, coefs, _ = enet_path(X, y)

# Plotting the Elastic Net path
plt.figure()
for coef, feature in zip(coefs, X.columns):
    plt.plot(alphas, coef, label=feature)

plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient')
plt.title('Elastic Net Path')
plt.legend()
plt.axis('tight')
plt.show()

# R-squared
r2 = r2_score(y_test, y_pred)
print(f"R-squared: {r2}")

# Adjusted R-squared
adj_r2 = calculate_adjusted_r_squared(r2, len(y_test), X_test.shape[1])
print(f"Adjusted R-squared: {adj_r2}")

# Jarque-Bera Test
jb_statistic, jb_p_value = stats.jarque_bera(residuals)
print(f"Jarque-Bera Test: Statistic: {jb_statistic}, p-value: {jb_p_value}")

# Breusch-Pagan Test
bp_test = het_breuschpagan(residuals, add_constant(X_train))
print(f"Breusch-Pagan: {bp_test}")