# Standard library imports
import os              # Provides a way of using operating system dependent functionality

# Third-party imports
import pandas as pd           # Data manipulation and analysis library
import numpy as np            # Scientific computing library
import matplotlib.pyplot as plt  # Plotting library for creating static and interactive visualizations
import seaborn as sns         # Data visualization library based on matplotlib
import statsmodels.api as sm  # Statistical models including OLS regression
from statsmodels.stats.outliers_influence import variance_inflation_factor  # VIF calculation
from statsmodels.stats.outliers_influence import OLSInfluence

# Get the absolute path to the directory of the current script
script_dir = os.path.abspath(os.path.dirname(__file__))

# Build the absolute path to the data file
data_path = os.path.join(script_dir, '..', '02_Exploratory_Data_Analysis', '2.5_transformed_melb_data.csv')

# Load the transformed Melbourne housing dataset
melb_data = pd.read_csv(data_path)

# Preparing the features (X) and target variable (y) for the regression model
X = melb_data.drop('Price_boxcox', axis=1)
X = sm.add_constant(X)  # Adding a constant term for the regression intercept
y = melb_data['Price_boxcox']

# Fitting the Ordinary Least Squares (OLS) regression model
model = sm.OLS(y, X).fit()

# Printing the summary of the regression model
print(model.summary())

fitted_values = model.fittedvalues
residuals = model.resid
influence = model.get_influence()
leverage = influence.hat_matrix_diag
cooks = influence.cooks_distance[0]

# Residuals vs Fitted Values Plot
plt.figure()
plt.scatter(fitted_values, residuals)
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('OLS, Residuals vs. Fitted Values')
plt.axhline(y=0, color='grey', linestyle='dashed')
plt.show()

# Q-Q Plot
fig = sm.qqplot(residuals, line='45', fit=True)
plt.title('Normal Q-Q')
plt.show()

# Scale-Location Plot (a.k.a. Spread-Location Plot)
plt.figure()
plt.scatter(fitted_values, np.sqrt(np.abs(residuals)))
plt.xlabel('Fitted Values')
plt.ylabel('Sqrt(Abs(Residuals))')
plt.title('Scale-Location Plot')
plt.show()

# Residuals vs Leverage Plot
plt.figure()
sm.graphics.influence_plot(model, criterion="cooks")
plt.title('Residuals vs Leverage')
plt.show()