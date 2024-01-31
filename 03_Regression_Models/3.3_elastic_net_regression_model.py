# Standard library imports
import os              # Provides a way of using operating system dependent functionality

# Third-party imports
import pandas as pd           # Data manipulation and analysis library
import numpy as np            # Scientific computing library
import matplotlib.pyplot as plt  # Plotting library for creating static and interactive visualizations
import seaborn as sns         # Data visualization library based on matplotlib
import statsmodels.api as sm  # Statistical models including OLS regression
from sklearn.preprocessing import StandardScaler  # StandardScaler for normalization
from sklearn.model_selection import train_test_split, RepeatedKFold, cross_val_score
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from statsmodels.stats.stattools import durbin_watson

# Get the absolute path to the directory of the current script
script_dir = os.path.abspath(os.path.dirname(__file__))

# Build the absolute path to the data file
data_path = os.path.join(script_dir, '..', '02_Exploratory_Data_Analysis', '2_transformed_melb_data.csv')

# Load the transformed Melbourne housing dataset
melb_data = pd.read_csv(data_path)

# Initialize the StandardScaler
scaler = StandardScaler()

# Applying StandardScaler to continuous variables
melb_data['Distance'] = scaler.fit_transform(melb_data[['Distance']])
melb_data['Landsize_no_outliers'] = scaler.fit_transform(melb_data[['Landsize_no_outliers']])

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
X = melb_data.drop('Price_log', axis=1)
y = melb_data['Price_log']

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

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
y_pred = elastic_net.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# * Model Diagnostics

# Residual Analysis 
residuals = y_test - y_pred
plt.scatter(y_pred, residuals)
plt.title('Residuals vs Predicted')
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.axhline(y=0, color='r', linestyle='-')
plt.show()

# Feature Importance
feature_importance = pd.DataFrame(model.coef_, index=X.columns, columns=['importance'])
print(feature_importance.sort_values(by='importance', ascending=False).to_string())

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print("Cross-validated scores:", scores)

# Diagnostic Plots (Q-Q Plot)
stats.probplot(residuals, dist="norm", plot=plt)
plt.show()

# Condition Number
coef_matrix = np.array([model.coef_])
condition_number = np.linalg.cond(coef_matrix)
print("Condition Number:", condition_number)

# Autocorrelation Check (Durbin-Watson Test)
dw = durbin_watson(residuals)
print("Durbin-Watson statistic:", dw)
