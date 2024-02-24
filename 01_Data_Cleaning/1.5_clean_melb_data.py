# Standard library imports
import sys # Provides a way of using operating system dependent functionality
import os  # For interacting with the operating system
from datetime import datetime  # For handling date and time data

# Third-party imports
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For creating visualizations
import seaborn as sns  # For high-level data visualization
import missingno as msno  # For visualizing missing data

# Define the absolute path of the parent directory of the script's grandparent directory
# This is useful for module importation from a different directory structure
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)  # Add it to the sys.path for module resolution

# Local application imports
from utils import plot_utils  # Importing visualization utilities

# * File Importation
# Determine the absolute path to the directory containing the current script
script_dir = os.path.abspath(os.path.dirname(__file__))

# Construct the path to the data file
data_path = os.path.join(script_dir, '..', '00_Raw_Data', '0_melb_data.csv')

# Load the dataset, parsing dates to ensure proper datetime format
melb_data = pd.read_csv(data_path, parse_dates=['Date'], dayfirst=True)

# Data manipulation: extracting year and month from the date for further analysis
melb_data['Year'] = melb_data['Date'].dt.year
melb_data['Month'] = melb_data['Date'].dt.month
melb_data['Date'] = melb_data['Date'].dt.date  # Simplify 'Date' to date format

# Remove duplicate entries to maintain data integrity
melb_data.drop_duplicates(inplace=True)

# * Pre-Cleaned Data Inspection
# Display summary statistics for the dataset
print(melb_data.describe())

# Identify missing values across columns
print(melb_data.isna().sum())

# Visualize missing data to guide cleaning and imputation strategies
msno.matrix(melb_data)
plt.show()

# * Data Preparation for Model Fitting
# Create binary indicators for specific conditions to enhance model interpretation
melb_data['Bathroom_was_0'] = np.where(melb_data['Bathroom'] == 0, True, False)
melb_data['Car_was_missing'] = melb_data['Car'].isna()
melb_data['Landsize_Indicator'] = np.where(melb_data['Landsize'] == 0, 1, 0)

# TODO: Create indicator variable for 'BuildingArea'

# Replace 0 bathroom counts with the median number of bathrooms.
melb_data['Bathroom'] = np.where(melb_data['Bathroom'] == 0, melb_data['Bathroom'].median(), melb_data['Bathroom'])

# Fill missing values in 'Car' column with the median number of cars.
melb_data.fillna(value={'Car': melb_data['Car'].median()}, inplace=True)

# TODO: Using Random Forest for Predictive Imputation for BuildingArea

# Define a function to determine the effective number of bedrooms
# based on 'Rooms' and 'Bedroom2' columns
def effective_bedrooms(row):
    """
    Calculate the effective number of bedrooms for a property.
    
    This function takes into account discrepancies between 'Rooms' and 
    'Bedroom2' columns. It applies logic to decide the most representative 
    value for the number of bedrooms.

    Parameters:
    row (pd.Series): A row of the DataFrame.

    Returns:
    int: The calculated effective number of bedrooms.
    """
    if row['Rooms'] > row['Bedroom2'] and row['Bedroom2'] != 0:
        return row['Bedroom2']
    elif row['Rooms'] < row['Bedroom2'] and row['Rooms'] != 0:
        return row['Rooms']
    elif row['Rooms'] == row['Bedroom2']:
        return row['Rooms']
    else:
        return row['Rooms'] + row['Bedroom2']

# Apply the effective_bedrooms function to the dataset
melb_data['NewBed'] = melb_data.apply(effective_bedrooms, axis=1)

def fill_councilarea(row):
    """
    Fill missing 'CouncilArea' based on properties in the same suburb.

    This function matches properties by 'Suburb', 'Postcode', 'Regionname', 
    and 'Propertycount' to fill missing 'CouncilArea' values. If a matching 
    property with a non-null 'CouncilArea' is found, its 'CouncilArea' value 
    is used.

    Parameters:
    row (pd.Series): A row from the DataFrame.

    Returns:
    str: The filled or original 'CouncilArea' value for the row.
    """
    if pd.isnull(row['CouncilArea']):
        match = melb_data[(row['Suburb'] == melb_data['Suburb']) & 
                          (row['Postcode'] == melb_data['Postcode']) & 
                          (row['Regionname'] == melb_data['Regionname']) & 
                          (row['Propertycount'] == melb_data['Propertycount']) & 
                          pd.notnull(melb_data['CouncilArea'])]
        if not match.empty:
            return match.iloc[0]['CouncilArea']
    return row['CouncilArea']

melb_data['CouncilArea'] = melb_data.apply(fill_councilarea, axis=1)

# Simplify the dataset by dropping columns not needed for further analysis
melb_data.drop(columns=['Rooms', 'Bedroom2', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount'], inplace=True)

# Reorder columns for readability and analysis purposes
melb_data = melb_data[[
    'Address', 'Postcode', 'Suburb', 'Regionname', 'CouncilArea', 'Type', 
    'Price', 'SellerG', 'Method', 'Date', 'Year', 'Month', 'NewBed', 
    'Bathroom', 'Bathroom_was_0', 'Car', 'Car_was_missing', 'Distance',
    'Landsize', 'Landsize_Indicator', 'BuildingArea'
]

# * Post-Cleaning Data Inspection
# Summarize the cleaned dataset to verify changes
print(melb_data.describe())

# Check for remaining missing values post-cleaning
print(melb_data.isna().sum())

# Visualize the cleaned data to ensure readiness for analysis
msno.matrix(melb_data)
plt.show()
print(melb_data.shape[0])

# Outlier handling: Identify and mitigate outliers in key quantitative columns
quan_columns = ['Price', 'NewBed', 'Bathroom', 'Car', 'Distance', 'Landsize']
plot_utils.plot_box(data=melb_data, column_list=quan_columns, rows=2, cols=3) # Utilize utility for boxplot visualization
    
# Identify and handle outliers for 'Landsize' column
q1, q3 = melb_data['Landsize'].quantile([0.25, 0.75])
IQR = q3 - q1
upper, lower = q3 + (1.5 * IQR), q1 - (1.5 * IQR)
melb_data = melb_data[(melb_data['Landsize'] >= lower) & (melb_data['Landsize'] <= upper)].copy()

# Re-visualize post-outlier adjustment to verify correction
plot_utils.plot_box(data=melb_data, column_list=quan_columns, rows=2, cols=3)
plt.show()
print(melb_data.shape[0])

# * Export Data
# Construct the full file path
output_file_path = os.path.join(script_dir, '1.5_cleaned_melb_data.csv')

# Export the cleaned and processed data to a new CSV file
melb_data.to_csv(output_file_path, index=False)  # Export without the index