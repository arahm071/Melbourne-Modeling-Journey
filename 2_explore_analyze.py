# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy


# Load dataset
melb_data = pd.read_csv('cleaned_melb_data.csv')

melb_columns = ['Price', 'Distance', 'NewBed', 'Bathroom', 'Car', 'Landsize']

def plot_skew(column_list, rows, cols, fig_x, fix_y):
    
    # Create a 3x2 subplot
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_x, fix_y))

    

# Create a 3x2 subplot
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))

# Plotting histogram for 'Price'
sns.histplot(data=melb_data, x='Price', ax=axs[0, 0])
axs[0, 0].set_title('Distribution of Property Prices')

# Plotting histogram for 'Distance'
sns.histplot(data=melb_data, x='Distance', ax=axs[0, 1])
axs[0, 1].set_title('Distribution of Distance from CBD')

# Plotting histogram for 'NewBed'
sns.histplot(data=melb_data, x='NewBed', ax=axs[0, 2])
axs[0, 2].set_title('Distribution of Number of Bedrooms')

# Plotting histogram for 'Bathroom'
sns.histplot(data=melb_data, x='Bathroom', ax=axs[1, 0])
axs[1, 0].set_title('Distribution of Number of Bathrooms')

# Plotting histogram for 'Car'
sns.histplot(data=melb_data, x='Car', ax=axs[1, 1])
axs[1, 1].set_title('Distribution of Car Parking Spaces')

# Plotting histogram for 'Landsize'
sns.histplot(data=melb_data, x='Landsize', ax=axs[1, 2])
axs[1, 2].set_title('Distribution of Landsize')

# Adjust layout so titles don't overlap with plots
plt.tight_layout()

# Show plot
plt.show()

print(melb_data[melb_columns].skew())

# Create a 3x2 subplot for boxplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Plotting boxplot for 'Price'
sns.boxplot(data=melb_data, x='Price', ax=axs[0, 0])
axs[0, 0].set_title('Boxplot of Property Prices')

# Plotting boxplot for 'Distance'
sns.boxplot(data=melb_data, x='Distance', ax=axs[0, 1])
axs[0, 1].set_title('Boxplot of Distance from CBD')

# Plotting boxplot for 'NewBed'
sns.boxplot(data=melb_data, x='NewBed', ax=axs[0, 2])
axs[0, 2].set_title('Boxplot of Number of Bedrooms')

# Plotting boxplot for 'Bathroom'
sns.boxplot(data=melb_data, x='Bathroom', ax=axs[1, 0])
axs[1, 0].set_title('Boxplot of Number of Bathrooms')

# Plotting boxplot for 'Car'
sns.boxplot(data=melb_data, x='Car', ax=axs[1, 1])
axs[1, 1].set_title('Boxplot of Car Parking Spaces')

# Plotting boxplot for 'Landsize'
sns.boxplot(data=melb_data, x='Landsize', ax=axs[1, 2])
axs[1, 2].set_title('Boxplot of Landsize')

# Adjust layout so titles don't overlap with plots
plt.tight_layout()

# Show plot
plt.show()


# Computing correlation matrix for numerical variables
correlation_matrix = melb_data[melb_columns].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix")  # Added title for clarity
plt.show() 

# Calculate the first quartile (Q1) and third quartile (Q3) for the current 'Landsize'
q1 = melb_data['Landsize'].quantile(0.25)
q3 = melb_data['Landsize'].quantile(0.75)

# Calculate the Interquartile Range (IQR) for the current 'Landsize'
IQR = q3 - q1

# Calculate the upper and lower bounds for outliers for the current 'Landsize'
upper = q3 + (1.5 * IQR)
lower = q1 - (1.5 * IQR)

# Filter outliers based on the upper and lower bounds for the current 'Landsize'
transformed_df = melb_data[(melb_data['Landsize'] >= lower) & (melb_data['Landsize'] <= upper)]
transformed_df = transformed_df.copy()  # Create a copy of the filtered DataFrame

# Log transformation for 'Price' and 'Bathroom' columns
transformed_df['Price'] = np.log(transformed_df['Price'])

transformed_df.rename(columns={"Price": "Price_log", "Landsize": "Landsize_no_outliers"}, inplace=True)

# Create a 3x2 subplot for histograms
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))

# Plotting histogram for 'Price_log'
sns.histplot(data=transformed_df, x='Price_log', ax=axs[0, 0])
axs[0, 0].set_title('Distribution of Log Transformed Property Prices')

# Plotting histogram for 'Distance' using transformed dataframe
sns.histplot(data=transformed_df, x='Distance', ax=axs[0, 1])
axs[0, 1].set_title('Distribution of Distance from CBD')

# Plotting histogram for 'NewBed' using transformed dataframe
sns.histplot(data=transformed_df, x='NewBed', ax=axs[0, 2])
axs[0, 2].set_title('Distribution of Number of Bedrooms')

# Plotting histogram for 'Bathroom' using transformed dataframe
sns.histplot(data=transformed_df, x='Bathroom', ax=axs[1, 0])
axs[1, 0].set_title('Distribution of Number of Bathrooms')

# Plotting histogram for 'Car' using transformed dataframe
sns.histplot(data=transformed_df, x='Car', ax=axs[1, 1])
axs[1, 1].set_title('Distribution of Car Parking Spaces')

# Plotting histogram for 'Landsize_no_outliers'
sns.histplot(data=transformed_df, x='Landsize_no_outliers', ax=axs[1, 2])
axs[1, 2].set_title('Distribution of Landsize with No Outliers')

# Adjust layout so titles don't overlap with plots
plt.tight_layout()

# Show plot
plt.show()

# Print skewness for the selected variables from the transformed dataframe
print(transformed_df[['Price_log', 'Distance', 'NewBed', 'Bathroom', 'Car', 'Landsize_no_outliers']].skew())

# Create a 3x2 subplot for boxplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))

# Plotting boxplot for 'Price_log'
sns.boxplot(data=transformed_df, x='Price_log', ax=axs[0, 0])
axs[0, 0].set_title('Boxplot of Log Transformed Property Prices')

# Plotting boxplot for 'Distance' using transformed dataframe
sns.boxplot(data=transformed_df, x='Distance', ax=axs[0, 1])
axs[0, 1].set_title('Boxplot of Distance from CBD')

# Plotting boxplot for 'NewBed' using transformed dataframe
sns.boxplot(data=transformed_df, x='NewBed', ax=axs[0, 2])
axs[0, 2].set_title('Boxplot of Number of Bedrooms')

# Plotting boxplot for 'Bathroom' using transformed dataframe
sns.boxplot(data=transformed_df, x='Bathroom', ax=axs[1, 0])
axs[1, 0].set_title('Boxplot of Number of Bathrooms')

# Plotting boxplot for 'Car' using transformed dataframe
sns.boxplot(data=transformed_df, x='Car', ax=axs[1, 1])
axs[1, 1].set_title('Boxplot of Car Parking Spaces')

# Plotting boxplot for 'Landsize_no_outliers'
sns.boxplot(data=transformed_df, x='Landsize_no_outliers', ax=axs[1, 2])
axs[1, 2].set_title('Boxplot of Landsize with No Outliers')

# Adjust layout so titles don't overlap with plots
plt.tight_layout()

# Show plot
plt.show()

correlation_matrix = transformed_df[['Price_log','Distance','NewBed','Bathroom','Car','Landsize_no_outliers']].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix After Transformations")  # Added title for clarity
plt.show()
