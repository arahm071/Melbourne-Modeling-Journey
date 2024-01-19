# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the transformed Melbourne housing dataset
melb_data = pd.read_csv('2_transformed_melb_data.csv')

# Creating dummy variables for categorical columns
dummies_Suburb = pd.get_dummies(melb_data['Suburb'], drop_first=True, dtype=int)
dummies_Regionname = pd.get_dummies(melb_data['Regionname'], drop_first=True, dtype=int)
dummies_Type = pd.get_dummies(melb_data['Type'], drop_first=True, dtype=int)
dummies_Method = pd.get_dummies(melb_data['Method'], drop_first=True, dtype=int)

# Selecting the top N suburbs for creating grouped suburb categories
num_top_suburbs = 50
top_suburbs = melb_data['Suburb'].value_counts().nlargest(num_top_suburbs).index
melb_data['Grouped_Suburb'] = melb_data['Suburb'].apply(lambda x: x if x in top_suburbs else 'Other')
dummies_Grouped_Suburb = pd.get_dummies(melb_data['Grouped_Suburb'], drop_first=True, dtype=int)

# Concatenating the dummy variables with the main DataFrame
melb_data = pd.concat([melb_data, dummies_Regionname, dummies_Type, dummies_Method, dummies_Grouped_Suburb], axis=1)

# Defining the columns to exclude from the regression model
excluded_columns = ['Address', 'Suburb', 'Regionname', 'Type', 'Price_log', 'Method', 'Date', 'Propertycount', 'Grouped_Suburb']

# Preparing the features (X) and target variable (y) for the regression model
X = melb_data[[x for x in melb_data.columns if x not in excluded_columns]]
X = sm.add_constant(X)
y = melb_data['Price_log']

# Fitting the OLS regression model and printing the summary
model = sm.OLS(y, X).fit()
print(model.summary())