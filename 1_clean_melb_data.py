# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Define a function to parse dates in the dataset
dateparse = lambda x: datetime.strptime(x, '%d/%m/%Y')

# Load dataset with custom date parsing
melb_data = pd.read_csv('melb_data.csv', parse_dates=['Date'], date_parser=dateparse)
melb_data['Date'] = melb_data['Date'].dt.date  # Convert datetime to date

# Remove duplicate entries
melb_data = melb_data.drop_duplicates()

# Fill missing 'Car' values with 0 (assuming missing values imply no car space)
melb_data = melb_data.fillna(value={'Car': 0})

# Function to determine the effective number of bedrooms
def effective_bedrooms(row):
    """
    Calculate the effective number of bedrooms.

    Takes the minimum of 'Rooms' and 'Bedroom2' unless one of them is 0,
    in which case it sums them up.

    Parameters:
    row (pd.Series): A row of the DataFrame.

    Returns:
    int: Effective number of bedrooms.
    """
    if row['Rooms'] > row['Bedroom2'] and row['Bedroom2'] != 0:
        return row['Bedroom2']
    elif row['Rooms'] < row['Bedroom2'] and row['Rooms'] != 0:
        return row['Rooms']
    elif row['Rooms'] == row['Bedroom2']:
        return row['Rooms']
    else:
        return row['Rooms'] + row['Bedroom2']

melb_data['NewBed'] = melb_data.apply(effective_bedrooms, axis=1)

# Assume at least one bathroom in every property
melb_data['Bathroom'] = np.where(melb_data['Bathroom'] == 0, 1, melb_data['Bathroom'])

# Drop columns with many missing values or those not needed for analysis
melb_data = melb_data.drop(columns=['SellerG', 'Postcode', 'CouncilArea', 'Lattitude', 'Longtitude', 'BuildingArea', 'YearBuilt', 'Rooms', 'Bedroom2'])
melb_data = melb_data[['Address', 'Suburb', 'Regionname', 'Type', 'Price', 'Method', 'Date', 'Distance', 'NewBed', 'Bathroom', 'Car', 'Landsize', 'Propertycount']]

# Export cleaned data to a new CSV file
melb_data.to_csv('cleaned_melb_data.csv', index=False)

# Uncomment below lines to explore the dataset
# print(melb_data.describe())
# print(melb_data[melb_data['Landsize'] == 0]['Type'].value_counts())

# Identify duplicate listings based on specific columns
columns_names = ['Address', 'Suburb', 'Propertycount']
duplicates = melb_data.duplicated(subset=columns_names, keep=False)
dup_melb_data = melb_data[duplicates].sort_values(by=['Address', 'Suburb', 'Date'])


