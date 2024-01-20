# Standard library imports
# (No standard library imports in this code)

# Third-party imports
import pandas as pd           # Data manipulation and analysis library
import numpy as np            # Scientific computing library
import matplotlib.pyplot as plt  # Plotting library for creating static and interactive visualizations
import seaborn as sns         # Data visualization library based on matplotlib

# Load the transformed Melbourne housing dataset
melb_data = pd.read_csv('1_clean_melb_data.csv')

# Identify and handle outliers for 'Landsize' column
q1 = melb_data['Landsize'].quantile(0.25)
q3 = melb_data['Landsize'].quantile(0.75)
IQR = q3 - q1
upper = q3 + (1.5 * IQR)
lower = q1 - (1.5 * IQR)
transformed_df = melb_data[(melb_data['Landsize'] >= lower) & (melb_data['Landsize'] <= upper)].copy()

# Rename columns to reflect transformations
transformed_df.rename(columns={"Landsize": "Landsize_no_outliers"}, inplace=True)
