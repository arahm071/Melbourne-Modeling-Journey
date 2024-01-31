# Standard library imports
import os              # Provides a way of using operating system dependent functionality

# Third-party imports
import pandas as pd              # Data manipulation and analysis library
import numpy as np               # Scientific computing library
import matplotlib.pyplot as plt  # Plotting library for creating static and interactive visualizations
import seaborn as sns            # Data visualization library based on matplotlib
import scipy                     # Library for scientific and technical computing

# Get the absolute path to the directory of the current script
script_dir = os.path.abspath(os.path.dirname(__file__))

# Build the absolute path to the data file
data_path = os.path.join(script_dir, '..', '01_Data_Cleaning', '1_cleaned_melb_data.csv')

# Load dataset containing cleaned Melbourne housing data
melb_data = pd.read_csv(data_path)

# Define columns for analysis
melb_columns = ['Price', 'Distance', 'NewBed', 'Bathroom', 'Car', 'Landsize']

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

# Perform initial plotting of data
plot_skew(data=melb_data, column_list=melb_columns, rows=2, cols=3)
plot_outliers(data=melb_data, column_list=melb_columns, rows=2, cols=3)

#Pairplot
sns.pairplot(data=melb_data[melb_columns])
plt.show()

# Compute and display correlation matrix for the numerical variables in the dataset
correlation_matrix = melb_data[melb_columns].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix")
plt.show()

# Identify and handle outliers for 'Landsize' column
q1 = melb_data['Landsize'].quantile(0.25)
q3 = melb_data['Landsize'].quantile(0.75)
IQR = q3 - q1
upper = q3 + (1.5 * IQR)
lower = q1 - (1.5 * IQR)
transformed_df = melb_data[(melb_data['Landsize'] >= lower) & (melb_data['Landsize'] <= upper)].copy()

# Apply log transformation to 'Price' column
transformed_df['Price'] = np.log(transformed_df['Price'])

# Rename columns to reflect transformations
transformed_df.rename(columns={"Price": "Price_log", "Landsize": "Landsize_no_outliers"}, inplace=True)

# Redefine columns for the transformed data
melb_transformed_columns = ['Price_log', 'Distance', 'NewBed', 'Bathroom', 'Car', 'Landsize_no_outliers']

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

#Pairplot (Which also contain Linearity Check (Don't use any catergorical variables))
sns.pairplot(data=transformed_df[melb_transformed_columns])
plt.show()

# Construct the full file path
output_file_path = os.path.join(script_dir, '2_transformed_melb_data.csv')

# Export transformed data to a new CSV file
transformed_df.to_csv(output_file_path, index=False)