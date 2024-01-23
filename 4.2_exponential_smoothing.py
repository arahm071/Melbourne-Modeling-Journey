# Standard library imports
# (No standard library imports in this code)

# Third-party imports
import pandas as pd           # Data manipulation and analysis library
import numpy as np            # Scientific computing library
import matplotlib.pyplot as plt  # Plotting library for creating static and interactive visualizations
import matplotlib.dates as mdates  # Handling date formats in matplotlib
import seaborn as sns         # Data visualization library based on matplotlib
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the transformed Melbourne housing dataset with dates parsed
melb_data = pd.read_csv('1_cleaned_melb_data.csv', parse_dates=['Date'])

# Setup data for time series analysis
# Sorting by date and aggregating average price
melb_data.sort_values(by='Date', inplace=True)
date_avgprice = melb_data.groupby('Date')['Price'].agg('mean').reset_index().set_index('Date')
date_avgprice.rename(columns={'mean': 'average_price'}, inplace=True)

df_resample = date_avgprice.resample('2W').mean()

df_resample.ffill(inplace=True)

split_point = int(len(df_resample)*0.8)

train_data, test_data = df_resample.iloc[:split_point], df_resample.iloc[split_point:]

# Initialize and fit the model
hw_model = ExponentialSmoothing(train_data, seasonal_periods=8, trend='add', seasonal='add').fit() 


