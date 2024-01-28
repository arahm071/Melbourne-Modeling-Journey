# Standard library imports
import os              # Provides a way of using operating system dependent functionality

# Third-party imports
import pandas as pd           # Data manipulation and analysis library
import numpy as np            # Scientific computing library
import matplotlib.pyplot as plt  # Plotting library for creating static and interactive visualizations
import seaborn as sns         # Data visualization library based on matplotlib
import statsmodels.api as sm  # Statistical models including OLS regression
from statsmodels.stats.outliers_influence import variance_inflation_factor  # VIF calculation
from sklearn.preprocessing import StandardScaler  # StandardScaler for normalization

# Get the absolute path to the directory of the current script
script_dir = os.path.abspath(os.path.dirname(__file__))

# Build the absolute path to the data file
data_path = os.path.join(script_dir, '..', '02_Data_Exploration', '2_transformed_melb_data.csv')

# Load the transformed Melbourne housing dataset
melb_data = pd.read_csv(data_path)

# Initialize the StandardScaler
scaler = StandardScaler()

# Applying StandardScaler to continuous variables
melb_data['Distance'] = scaler.fit_transform(melb_data[['Distance']])
melb_data['Landsize_no_outliers'] = scaler.fit_transform(melb_data[['Landsize_no_outliers']])

# Creating dummy variables for categorical columns
# ! dummies_Suburb = pd.get_dummies(melb_data['Suburb'], drop_first=True, dtype=int)
dummies_Regionname = pd.get_dummies(melb_data['Regionname'], drop_first=True, dtype=int)
dummies_Type = pd.get_dummies(melb_data['Type'], drop_first=True, dtype=int)
dummies_Method = pd.get_dummies(melb_data['Method'], drop_first=True, dtype=int)

# ! Uncomment the following block if you want to use grouped suburb categories
'''
num_top_suburbs = 10
top_suburbs = melb_data['Suburb'].value_counts().nlargest(num_top_suburbs).index
Grouped_Suburb = melb_data['Suburb'].apply(lambda x: x if x in top_suburbs else 'Other')
dummies_Grouped_Suburb = pd.get_dummies(Grouped_Suburb, drop_first=True, dtype=int)
'''

# Concatenating the dummy variables with the main DataFrame
melb_data = pd.concat([melb_data, dummies_Regionname, dummies_Type, dummies_Method], axis=1)

# Dropping columns not used in the regression model
excluded_columns = ['Address', 'Suburb', 'Regionname', 'Type', 'Method', 'Date', 'Propertycount']
melb_data.drop(excluded_columns, axis=1, inplace=True)

# Preparing the features (X) and target variable (y) for the regression model
X = melb_data.drop('Price_log', axis=1)
X = sm.add_constant(X)  # Adding a constant term for the regression intercept
y = melb_data['Price_log']

# Fitting the Ordinary Least Squares (OLS) regression model
model = sm.OLS(y, X).fit()

# Printing the summary of the regression model
print(model.summary())

# ! Detecting Multicollinearity with VIF (For high condition number)
'''
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True)
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)  # vif_data shows all VIFs are below the common threshold, so multicollinearity is not a significant concern
'''