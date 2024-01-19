# Standard library imports
from datetime import datetime  # For handling date and time data

# Third-party imports
import pandas as pd            # Popular data manipulation and analysis library for Python
import numpy as np             # Fundamental package for scientific computing in Python
import matplotlib.pyplot as plt # Plotting library for creating static, animated, and interactive visualizations
import seaborn as sns          # Data visualization library based on matplotlib

# Define a lambda function for date parsing in the dataset
# This is used to convert date strings into datetime objects
dateparse = lambda x: datetime.strptime(x, '%d/%m/%Y')

# Load the dataset with custom date parsing for the 'Date' column
# and immediately convert datetime objects to date format
melb_data = pd.read_csv('0_melb_data.csv', parse_dates=['Date'], date_parser=dateparse)
melb_data['Date'] = melb_data['Date'].dt.date

# Remove duplicate entries from the dataset to ensure data integrity
melb_data.drop_duplicates(inplace=True)


# Fill missing values in the 'Car' column with 0
# Assuming that missing values imply no car space available
melb_data.fillna(value={'Car': 0}, inplace=True)

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

# Ensure that every property is listed with at least one bathroom
melb_data['Bathroom'] = np.where(melb_data['Bathroom'] == 0, 1, melb_data['Bathroom'])

# Drop columns that are not needed for further analysis or have many missing values
melb_data.drop(columns=['SellerG', 'Postcode', 'CouncilArea', 
                        'Lattitude', 'Longtitude', 'BuildingArea', 
                        'YearBuilt', 'Rooms', 'Bedroom2'], inplace=True)


# Reorder columns for a more organized DataFrame
melb_data = melb_data[['Address', 'Suburb', 'Regionname', 'Type', 'Price', 'Method', 
                       'Date', 'Distance', 'NewBed', 'Bathroom', 'Car', 
                       'Landsize', 'Propertycount']]

# Export the cleaned and processed data to a new CSV file
melb_data.to_csv('1_cleaned_melb_data.csv', index=False)

# Uncomment the following lines to explore the dataset
# print(melb_data.describe())
# print(melb_data[melb_data['Landsize'] == 0]['Type'].value_counts())

# Identify and sort duplicate listings based on specific columns
columns_names = ['Address', 'Suburb', 'Propertycount']
duplicates = melb_data.duplicated(subset=columns_names, keep=False)
dup_melb_data = melb_data[duplicates].sort_values(by=['Address', 'Suburb', 'Date'])
