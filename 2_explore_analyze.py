# Importing necessary libraries for data analysis and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# Load dataset containing Melbourne housing data
melb_data = pd.read_csv('cleaned_melb_data.csv')
melb_columns = ['Price', 'Distance', 'NewBed', 'Bathroom', 'Car', 'Landsize']

def plot_skew(data, column_list, rows, cols, fig_x=15, fig_y=15):
    """
    Plots histograms for each specified column in a dataset.

    Args:
    data: DataFrame to plot data from.
    column_list: List of column names to plot histograms for.
    rows: Number of subplot rows.
    cols: Number of subplot columns.
    fig_x: Figure width.
    fig_y: Figure height.
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
    Plots boxplots for each specified column in a dataset to show outliers.

    Args:
    data: DataFrame to plot data from.
    column_list: List of column names to plot boxplots for.
    rows: Number of subplot rows.
    cols: Number of subplot columns.
    fig_x: Figure width.
    fig_y: Figure height.
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

# Plot histograms and boxplots for the original data
plot_skew(data=melb_data, column_list=melb_columns, rows=2, cols=3)
plot_outliers(data=melb_data, column_list=melb_columns, rows=2, cols=3)

# Compute and display correlation matrix for numerical variables
correlation_matrix = melb_data[melb_columns].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix")  # Add title for clarity
plt.show()

# Calculate and filter outliers for 'Landsize' column
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

# Columns for the transformed data
melb_transformed_columns = ['Price_log', 'Distance', 'NewBed', 'Bathroom', 'Car', 'Landsize_no_outliers']

# Plot histograms and boxplots for the transformed data
plot_skew(data=transformed_df, column_list=melb_transformed_columns, rows=2, cols=3)
plot_outliers(data=transformed_df, column_list=melb_transformed_columns, rows=2, cols=3)

# Print skewness for the selected variables from the transformed dataframe
print(transformed_df[melb_transformed_columns].skew())

# Compute and display correlation matrix for the transformed data
correlation_matrix_transformed = transformed_df[melb_transformed_columns].corr()
sns.heatmap(correlation_matrix_transformed, annot=True)
plt.title("Correlation Matrix After Transformations")  # Added title for clarity
plt.show()

# Export transformed data to a new CSV file
transformed_df.to_csv('transformed_melb_data.csv', index=False)
