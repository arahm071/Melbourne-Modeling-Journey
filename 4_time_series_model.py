# Standard library imports
# (No standard library imports in this code)

# Third-party imports
import pandas as pd           # Data manipulation and analysis library
import numpy as np            # Scientific computing library
import matplotlib.pyplot as plt  # Plotting library for creating static and interactive visualizations
import seaborn as sns         # Data visualization library based on matplotlib

# Load the transformed Melbourne housing dataset
melb_data = pd.read_csv('1_cleaned_melb_data.csv', parse_dates=['Date'])

#Setup data for timeseries use
melb_data.sort_values(by='Date', inplace=True)
date_avgprice = melb_data.groupby('Date')['Price'].mean().reset_index().set_index('Date')