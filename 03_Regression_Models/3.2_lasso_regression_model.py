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
from sklearn.linear_model import Lasso, LassoCV, lasso_path
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
# Define a cross-validation strategy for LassoCV
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# Initialize the LassoCV object with a range of alpha values and the defined CV strategy
lasso_cv = LassoCV(alphas=np.arange(0.00001, 0.2, 0.0001), cv=cv, n_jobs=-1)

# Fit the LassoCV model on the training data
lasso_cv.fit(X_train, y_train)

# Output the optimal alpha value determined by cross-validation
print(f"Optimal alpha: {lasso_cv.alpha_}")
'''

# Initialize the Lasso regression model with an alpha value
lasso = Lasso(alpha=0.00001)

# Fitting the model
model = lasso.fit(X_train, y_train)

# Evaluating the model
train_score = lasso.score(X_train, y_train)
test_score = lasso.score(X_test, y_test)
coeff_used = np.sum(lasso.coef_ != 0)

print(f"Training score: {train_score}")
print(f"Test score: {test_score}")
print(f"Number of features used: {coeff_used}")

# Predicting on the test data
y_pred = lasso.predict(X_test)

# Calculating the mean squared error
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

# Compute the Lasso path with lasso_path function
alphas, coefs, _ = lasso_path(X, y)

# Plotting the Lasso path
plt.figure()
for coef, feature in zip(coefs, X.columns):
    plt.plot(alphas, coef, label=feature)

plt.xscale('log')
plt.xlabel('Alpha')
plt.ylabel('Coefficient')
plt.title('Lasso Path')
plt.legend()
plt.axis('tight')
plt.show()

'''
# External Validation
# Predict on validation data and evaluate the model
y_val_pred = model.predict(X_val)
print("Validation score:", model.score(X_val, y_val))
'''