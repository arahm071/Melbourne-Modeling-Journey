# Standard library imports
import os              # Provides a way of using operating system dependent functionality

# Third-party imports
import pandas as pd              # Data manipulation and analysis library
import numpy as np               # Scientific computing library
import matplotlib.pyplot as plt  # Plotting library for creating static and interactive visualizations
import seaborn as sns            # Data visualization library based on matplotlib
from scipy import stats                         # Library for scientific and technical computing
from sklearn.preprocessing import MinMaxScaler  # Feature scaling library

# * File Importation

# Get the absolute path to the directory of the current script
script_dir = os.path.abspath(os.path.dirname(__file__))

# Build the absolute path to the data file
data_path = os.path.join(script_dir, '..', '01_Data_Cleaning', '1_cleaned_melb_data.csv')

# Load dataset containing cleaned Melbourne housing data
melb_data = pd.read_csv(data_path)

# Define columns for analysis
melb_columns = ['Price', 'Distance', 'NewBed', 'Bathroom', 'Car', 'Landsize']

# * Setup For Pre-Analysis Visualization

def plot_skew(data, column_list, rows, cols, fig_x=15, fig_y=15):
    """
    Plots histograms for each specified column in a dataset to analyze skewness.

    Parameters:
    data (DataFrame): The dataset to plot.
    column_list (list): List of column names to plot histograms for.
    rows (int): Number of rows in the subplot grid.
    cols (int): Number of columns in the subplot grid.
    fig_x (int): Width of the figure.
    fig_y (int): Height of the figure.
    """
    # Create a subplot grid
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_x, fig_y))
    rows_count, cols_count = 0, 0

    # Iterate through the column list and plot histograms
    for column_name in column_list:
        sns.histplot(data=data, x=column_name, ax=axs[rows_count, cols_count])
        axs[rows_count, cols_count].set_title(f'Distribution of {column_name}')
        cols_count += 1

        # Move to the next row if the current row is filled
        if cols_count >= cols:
            cols_count = 0
            rows_count += 1

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plot

def plot_outliers(data, column_list, rows, cols, fig_x=15, fig_y=15):
    """
    Plots boxplots for each specified column in a dataset to identify outliers.

    Parameters:
    data (DataFrame): The dataset to plot.
    column_list (list): List of column names to plot boxplots for.
    rows (int): Number of rows in the subplot grid.
    cols (int): Number of columns in the subplot grid.
    fig_x (int): Width of the figure.
    fig_y (int): Height of the figure.
    """
    # Create a subplot grid
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_x, fig_y))
    rows_count, cols_count = 0, 0

    # Iterate through the column list and plot boxplots
    for column_name in column_list:
        sns.boxplot(data=data, x=column_name, ax=axs[rows_count, cols_count])
        axs[rows_count, cols_count].set_title(f'Boxplot of {column_name}')
        cols_count += 1

        # Move to the next row if the current row is filled
        if cols_count >= cols:
            cols_count = 0
            rows_count += 1

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()  # Display the plot

# * Pre-Analysis

# Perform initial plotting of data
plot_skew(data=melb_data, column_list=melb_columns, rows=2, cols=3)
plot_outliers(data=melb_data, column_list=melb_columns, rows=2, cols=3)

# Print skewness for selected variables from the melb_data dataframe
print(melb_data[melb_columns].skew())

# Generating pairplot for selected variables to visualize distributions and relationships
sns.pairplot(data=melb_data[melb_columns])
plt.show()  # Display the pairplot

# Compute and display correlation matrix for the numerical variables in the dataset
correlation_matrix = melb_data[melb_columns].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix")
plt.show()

#Descriptive statistics of melb_data
print(melb_data.describe())

# * Data Transformation Process

# Identify and handle outliers for 'Landsize' column
q1 = melb_data['Landsize'].quantile(0.25)
q3 = melb_data['Landsize'].quantile(0.75)
IQR = q3 - q1
upper = q3 + (1.5 * IQR)
lower = q1 - (1.5 * IQR)
transformed_df = melb_data[(melb_data['Landsize'] >= lower) & (melb_data['Landsize'] <= upper)].copy()

# Creating an indicator variable for properties with no land size listed
transformed_df['Landsize_Indicator'] = np.where(transformed_df['Landsize'] == 0, 1, 0)

# Applying transformations to reduce skewness and normalize distributions
# Box-Cox transformation for 'Price' to address skewness
transformed_df['Price'], fitted_lambda = stats.boxcox(transformed_df['Price'])

# Yeo-Johnson transformation for 'Distance', 'NewBed', and 'Car' to normalize data
transformed_df['Distance'], fitted_lambda = stats.yeojohnson(transformed_df['Distance'])
transformed_df['NewBed'], fitted_lambda = stats.yeojohnson(transformed_df['NewBed'])
transformed_df['Car'], fitted_lambda = stats.yeojohnson(transformed_df['Car'])

# Box-Cox transformation for 'Bathroom' to normalize data
transformed_df['Bathroom'], fitted_lambda = stats.boxcox(transformed_df['Bathroom'])

# Renaming columns to reflect the type of transformations applied
transformed_df.rename(columns={
    "Price": "Price_boxcox", 
    "Distance": "Distance_yeojohnson",
    "NewBed": "NewBed_yeojohnson",
    "Bathroom": "Bathroom_boxcox",
    "Car": "Car_yeojohnson",
    "Landsize": "Landsize_no_out"
}, inplace=True)

# Create an instance of MinMaxScaler
scaler = MinMaxScaler()

# Fit the scaler to the data and transform the data
transformed_df['Distance_yeojohnson'] = scaler.fit_transform(transformed_df[['Distance_yeojohnson']])
transformed_df['Landsize_no_out'] = scaler.fit_transform(transformed_df[['Landsize_no_out']])

# Redefine columns for the transformed data
melb_transformed_columns = ['Price_boxcox', 'Distance_yeojohnson', 'NewBed_yeojohnson', 'Bathroom_boxcox', 'Car_yeojohnson', 'Landsize_no_out']

# * Post-Analysis

# Plot histograms and boxplots for the transformed data
plot_skew(data=transformed_df, column_list=melb_transformed_columns, rows=2, cols=3)
plot_outliers(data=transformed_df, column_list=melb_transformed_columns, rows=2, cols=3)

# Print skewness for selected variables from the transformed dataframe
print(transformed_df[melb_transformed_columns].skew())

# Compute and display correlation matrix for the transformed data
correlation_matrix_transformed = transformed_df[melb_transformed_columns].corr()
sns.heatmap(correlation_matrix_transformed, annot=True)
plt.title("Correlation Matrix After Transformations")
plt.show()

# Generating pairplot for the transformed variables to visualize distributions, relationships,
# and conduct a basic linearity check (excluding categorical variables)
sns.pairplot(data=transformed_df[melb_transformed_columns])
plt.show()  # Display the pairplot

#Descriptive statistics of transformed_df
print(transformed_df.describe())

# * File Exportation

# Construct the full file path
output_file_path = os.path.join(script_dir, '2_transformed_melb_data.csv')

# Export transformed data to a new CSV file
transformed_df.to_csv(output_file_path, index=False)