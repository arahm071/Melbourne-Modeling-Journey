# Standard library imports
# (No standard library imports in this code)

# Third-party imports
import pandas as pd           # Data manipulation and analysis library
import numpy as np            # Scientific computing library
import matplotlib.pyplot as plt  # Plotting library for creating static and interactive visualizations
import matplotlib.dates as mdates  # Handling date formats in matplotlib
import seaborn as sns         # Data visualization library based on matplotlib
from statsmodels.tsa.holtwinters import ExponentialSmoothing  # Exponential smoothing models
from sklearn.model_selection import TimeSeriesSplit  # Time series cross-validator
from sklearn.metrics import mean_squared_error  # Metric for error calculation

# Load the transformed Melbourne housing dataset with dates parsed
melb_data = pd.read_csv('1_cleaned_melb_data.csv', parse_dates=['Date'])

# Setup data for time series analysis
# Sorting by date and aggregating average price
melb_data.sort_values(by='Date', inplace=True)
date_avgprice = melb_data.groupby('Date')['Price'].agg('mean').reset_index().set_index('Date')
date_avgprice.rename(columns={'Price': 'average_price'}, inplace=True)

# Resampling data bi-weekly and forward filling missing values
df_resample = date_avgprice.resample('2W').mean()
df_resample.ffill(inplace=True)

'''
# Spliting data for timeseries model
split_point = int(len(df_resample) * 0.8)
train_data, test_data = df_resample.iloc[:split_point], df_resample.iloc[split_point:]
'''


# * Optimal parameters search for Exponential Smoothing

# Define your grid of parameters to search over
param_grid = {
    'trend': ['add', 'mul', None],
    'seasonal': ['add', 'mul', None],
    'seasonal_periods': [6],  # Adjust based on the seasonality of your data
    'smoothing_level': [0.1, 0.3, 0.5, 0.7, 0.9],
    'smoothing_slope': [0.1, 0.3, 0.5, 0.7, 0.9],
    'smoothing_seasonal': [0.1, 0.3, 0.5, 0.7, 0.9]
}

# Initialize variables for Grid Search
best_score = float('inf')
best_params = {}

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=3)

# Extract the 'average_price' for fitting
data = df_resample['average_price']

# Iterate over each combination of parameters
for trend in param_grid['trend']:
    for seasonal in param_grid['seasonal']:
        for periods in param_grid['seasonal_periods']:
            for alpha in param_grid['smoothing_level']:
                for beta in param_grid['smoothing_slope']:
                    for gamma in param_grid['smoothing_seasonal']:
                        mse_scores = []

                        for train_index, val_index in tscv.split(data):
                            train, val = data.iloc[train_index], data.iloc[val_index]

                            # Try/Except block is used to handle cases where the parameter combination is invalid
                            try:
                                model = ExponentialSmoothing(
                                    train,
                                    trend=trend,
                                    seasonal=seasonal,
                                    seasonal_periods=periods
                                ).fit(
                                    smoothing_level=alpha,
                                    smoothing_slope=beta,
                                    smoothing_seasonal=gamma,
                                    optimized=False
                                )
                                
                                predictions = model.forecast(len(val))
                                mse = mean_squared_error(val, predictions)
                                mse_scores.append(mse)
                            except Exception as e:
                                print(f'Error with combination: {e}')
                                break  # Break out of the loop if the parameter combination is invalid

                        # Compute the average MSE for the current parameter combination
                        if mse_scores:
                            avg_mse = np.mean(mse_scores)
                            print(f'Trend: {trend}, Seasonal: {seasonal}, Periods: {periods}, Alpha: {alpha}, Beta: {beta}, Gamma: {gamma}, MSE: {avg_mse}')

                            # Update the best score and parameters if current MSE is lower
                            if avg_mse < best_score:
                                best_score = avg_mse
                                best_params = {
                                    'trend': trend,
                                    'seasonal': seasonal,
                                    'seasonal_periods': periods,
                                    'smoothing_level': alpha,
                                    'smoothing_slope': beta,
                                    'smoothing_seasonal': gamma
                                }

# Display the best set of parameters
print(f'Best Score (MSE): {best_score}')
print('Best Parameters:')
print(best_params)

'''
# Initialize and fit the model
hw_model = ExponentialSmoothing(train_data, seasonal_periods=6, trend=None, seasonal=None).fit(smoothing_level=0.3, smoothing_slope=0.1, smoothing_seasonal=0.1, optimized=False)
print(hw_model.summary())
'''
