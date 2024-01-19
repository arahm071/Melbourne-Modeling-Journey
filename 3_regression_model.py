#Import stuff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load dataset containing transformed Melbourne housing data
melb_data = pd.read_csv('2_transformed_melb_data.csv')


#Dummy Catergorical Columns
dummy_Suburb = pd.get_dummies(melb_data['Suburb'], drop_first=True, dtype=int)
dummy_Regionname = pd.get_dummies(melb_data['Regionname'], drop_first=True, dtype=int)
dummy_Type = pd.get_dummies(melb_data['Type'], drop_first=True, dtype=int)
dummy_Method = pd.get_dummies(melb_data['Method'], drop_first=True, dtype=int)

#Dummy Suburbs Column
N = 50

top_suburbs = melb_data['Suburb'].value_counts().nlargest(N).index

melb_data['Suburb_Grouped'] = melb_data['Suburb'].apply(lambda x: x if x in top_suburbs else 'Other')

dummy_Suburb_Grouped = pd.get_dummies(melb_data['Suburb_Grouped'], drop_first=True, dtype=int)


#Join Dummies columns to main df
melb_data = pd.concat([melb_data, dummy_Regionname, dummy_Type, dummy_Method], axis=1)

#skip columns in list
skip_column = ['Address', 'Suburb', 'Regionname', 'Type', 'Price_log', 'Method', 'Date', 'Propertycount', 'Suburb_Grouped']

#Regression Model
X = melb_data[[x for x in melb_data.columns if x not in skip_column]]

X = sm.add_constant(X)

y = melb_data['Price_log']

model = sm.OLS(y,X).fit()

print(model.summary())