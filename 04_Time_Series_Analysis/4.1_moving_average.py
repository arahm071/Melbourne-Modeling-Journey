# Standard library imports
import os              # Provides a way of using operating system dependent functionality

# Third-party imports
import pandas as pd           # Data manipulation and analysis library
import numpy as np            # Scientific computing library
import matplotlib.pyplot as plt  # Plotting library for creating static and interactive visualizations
import matplotlib.dates as mdates  # Handling date formats in matplotlib
import seaborn as sns         # Data visualization library based on matplotlib

# Get the absolute path to the directory of the current script
script_dir = os.path.abspath(os.path.dirname(__file__))

# Build the absolute path to the data file
data_path = os.path.join(script_dir, '..', '01_Data_Cleaning', '1.5_cleaned_melb_data.csv')

# Load the cleaned Melbourne housing dataset with dates parsed
melb_data = pd.read_csv(data_path, parse_dates=['Date'])

# Setup data for time series analysis
# Sorting by date and aggregating average price and count of houses
melb_data.sort_values(by='Date', inplace=True)
date_avgprice = melb_data.groupby('Date')['Price'].agg(['mean', 'count']).reset_index().set_index('Date')
date_avgprice.rename(columns={'mean': 'average_price', 'count': 'num_of_houses'}, inplace=True)

# Calculating rolling average price over a 7-day window
date_avgprice['rolling_average_price'] = date_avgprice['average_price'].rolling(7).mean()

# Plotting Average Price, Rolling Average Price, and Total Number of Houses Sold vs Date
plt.figure(figsize=(12, 6))

# Format the date axis for better readability
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Plot average and rolling average prices
sns.lineplot(data=date_avgprice, x='Date', y='average_price', label='Average Price', linestyle='-')
sns.lineplot(data=date_avgprice, x='Date', y='rolling_average_price', label='Rolling Average Price', linestyle='--')

# Creating a second Y-axis for number of houses sold
ax2 = plt.gca().twinx()

# Plot number of houses sold on the secondary Y-axis
sns.lineplot(data=date_avgprice, x='Date', y='num_of_houses', label='Number of Houses Sold', color='g', ax=ax2, marker='o', alpha=0.3)

# Setting up legends and labels
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0)
ax2.legend(loc='upper right', bbox_to_anchor=(0.997, 0.94), borderaxespad=0)
plt.title('Average Price, Rolling Average Price and Number of Houses vs Date')
plt.xlabel('Date')
plt.ylabel('Price')
ax2.set_ylabel('Number of Houses Sold')

# Display the plot
plt.show()