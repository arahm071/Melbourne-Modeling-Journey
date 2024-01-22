# Standard library imports
# (No standard library imports in this code)

# Third-party imports
import pandas as pd           # Data manipulation and analysis library
import numpy as np            # Scientific computing library
import matplotlib.pyplot as plt  # Plotting library for creating static and interactive visualizations
import matplotlib.dates as mdates
import seaborn as sns         # Data visualization library based on matplotlib

# Load the transformed Melbourne housing dataset
melb_data = pd.read_csv('1_cleaned_melb_data.csv', parse_dates=['Date'])

#Setup data for timeseries use
melb_data.sort_values(by='Date', inplace=True)
date_avgprice = melb_data.groupby('Date')['Price'].agg(['mean','count']).reset_index().set_index('Date')
date_avgprice.rename(columns={'mean': 'average_price', 'count':'num_of_houses'}, inplace=True)

# * Rolling Average
date_avgprice['rolling_average_price'] = date_avgprice['average_price'].rolling(7).mean()


#Plotting Average price, rolling average price and total # of houses sold vs Date
plt.figure(figsize=(12,6))

# Format the date axis
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

# Plot the first two line plots with label
line1 = sns.lineplot(data=date_avgprice, x='Date', y='average_price', label='Average Price', linestyle='-')
line2 = sns.lineplot(data=date_avgprice, x='Date', y='rolling_average_price', label='Rolling Average Price', linestyle='--')

# Create a second Y-axis for the third line plot
ax2 = plt.gca().twinx()

# Plot the third line plot on the second Y-axis with label
line3 = sns.lineplot(data=date_avgprice, x='Date', y='num_of_houses', label='Number of Houses Sold', color='g', ax=ax2, marker='o', alpha=0.3)

# Set the first legend with bbox_to_anchor to place it outside of the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

# Set the second legend for the twin axis and adjust as necessary
ax2.legend(loc='upper right', bbox_to_anchor=(0.997, 0.94), borderaxespad=0.)

# Set the title and labels
plt.title('Average Price, Rolling Average Price and Number of Houses vs Date')
plt.xlabel('Date')
plt.ylabel('Price')
ax2.set_ylabel('Number of Houses Sold')

plt.show()