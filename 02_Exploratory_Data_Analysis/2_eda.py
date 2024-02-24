# Standard library imports
import sys
import os  # Provides OS dependent functionality, such as file path manipulations

# Third-party imports
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For scientific computing and array objects
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
import seaborn as sns  # For data visualization based on matplotlib
from scipy import stats  # For scientific and technical computing
from sklearn.preprocessing import MinMaxScaler  # For feature scaling

# Adding the parent directory of this script to sys.path for module importation
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Local application imports
from utils import plot_utils  # Importing utility functions for plotting

# * File Importation

# Get the absolute path to the directory of the current script
script_dir = os.path.abspath(os.path.dirname(__file__))

# Build the absolute path to the data file
data_path = os.path.join(script_dir, '..', '01_Data_Cleaning', '1_cleaned_melb_data.csv')

# Load dataset containing cleaned Melbourne housing data
melb_data = pd.read_csv(data_path)

# * Pre-Analysis

# Define quantitative and categorical columns for subsequent analysis
quan_columns = ['Price', 'NewBed', 'Bathroom', 'Car', 'Distance', 'Landsize']
cat_columns = ['Postcode', 'Suburb', 'Regionname', 'CouncilArea', 'Type', 'SellerG', 'Method', 'Year', 'Month']

# Initial plotting to understand data distributions
plot_utils.plot_hist(data=melb_data, column_list=quan_columns, rows=2, cols=3)
plot_utils.plot_qq(data=melb_data, column_list=quan_columns, rows=2, cols=3)
plot_utils.plot_box(data=melb_data, column_list=quan_columns, rows=2, cols=3)
plot_utils.plot_box(data=melb_data, column_list=cat_columns, price=True, rows=3, cols=3)

# Assess skewness of quantitative variables
print(melb_data[quan_columns].skew())

# Visualize pair-wise relationships to identify potential correlations and trends
sns.pairplot(data=melb_data[quan_columns])
plt.show()  # Display the pairplot

# Display correlation matrix to assess linear relationships between variables
correlation_matrix = melb_data[quan_columns].corr()
sns.heatmap(correlation_matrix, annot=True)
plt.title("Correlation Matrix")
plt.show()

# Display descriptive statistics to summarize central tendency, dispersion, and shape
print(melb_data.describe())

# * Feature Engineering
# Copy the DataFrame to apply transformations without altering the original data
melb_fe = melb_data.copy()

# Creating dummy variables for categorical features
# Note: 'drop_first=True' avoids dummy variable trap by removing the first level
dummies_Postcode = pd.get_dummies(melb_fe['Postcode'], drop_first=True, dtype=int)
dummies_Suburb = pd.get_dummies(melb_fe['Suburb'], drop_first=True, dtype=int)
dummies_SellerG = pd.get_dummies(melb_fe['SellerG'], drop_first=True, dtype=int)
dummies_Regionname = pd.get_dummies(melb_fe['Regionname'], drop_first=True, dtype=int)
dummies_CouncilArea = pd.get_dummies(melb_fe['CouncilArea'], drop_first=True, dtype=int)
dummies_Type = pd.get_dummies(melb_fe['Type'], drop_first=True, dtype=int)
dummies_Method = pd.get_dummies(melb_fe['Method'], drop_first=True, dtype=int)
dummies_Year = pd.get_dummies(melb_fe['Year'], drop_first=True, dtype=int)
dummies_Month = pd.get_dummies(melb_fe['Month'], drop_first=True, dtype=int)

# ! Uncomment the following block if you want to use grouped suburb categories
'''
num_top_suburbs = 10
top_suburbs = melb_data['Suburb'].value_counts().nlargest(num_top_suburbs).index
Grouped_Suburb = melb_data['Suburb'].apply(lambda x: x if x in top_suburbs else 'Other')
dummies_Grouped_Suburb = pd.get_dummies(Grouped_Suburb, drop_first=True, dtype=int)
'''

# Concatenate dummy variables with the main DataFrame, dropping original categorical columns
melb_fe = pd.concat([melb_fe, dummies_Suburb, dummies_SellerG, dummies_Postcode, dummies_Regionname, dummies_CouncilArea, dummies_Type, dummies_Method, dummies_Year, dummies_Month], axis=1)  # Concatenating along columns
excluded_columns = ['Address', 'Postcode', 'Suburb', 'Regionname', 'CouncilArea', 'Type', 
                    'SellerG', 'Method', 'Date', 'Year', 'Month', 'BuildingArea'] # List columns to drop
melb_fe.drop(excluded_columns, axis=1, inplace=True)

# Construct the full file path
# ! output_file_path_1 = os.path.join(script_dir, '2_untransformed_melb_data.csv')

# Export transformed data to a new CSV file
# ! melb_fe.to_csv(output_file_path_1, index=False)

# TODO: Polynomial Features

# Applying transformations to reduce skewness and normalize distributions
# Box-Cox transformation for 'Price' to address skewness
melb_fe['Price'], fitted_lambda = stats.boxcox(melb_fe['Price'])

# Yeo-Johnson transformation for 'Distance', 'NewBed', and 'Car' to normalize data
melb_fe['Distance'], fitted_lambda = stats.yeojohnson(melb_fe['Distance'])
melb_fe['NewBed'], fitted_lambda = stats.yeojohnson(melb_fe['NewBed'])
melb_fe['Car'], fitted_lambda = stats.yeojohnson(melb_fe['Car'])

# Box-Cox transformation for 'Bathroom' to normalize data
melb_fe['Bathroom'], fitted_lambda = stats.boxcox(melb_fe['Bathroom'])

# Renaming columns to reflect the type of transformations applied
melb_fe.rename(columns={
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
melb_fe['Distance_yeojohnson'] = scaler.fit_transform(melb_fe[['Distance_yeojohnson']])
melb_fe['Landsize_no_out'] = scaler.fit_transform(melb_fe[['Landsize_no_out']])

# TODO: Interaction terms

# Redefine columns for the transformed data
transformed_quan = ['Price_boxcox', 'Distance_yeojohnson', 'NewBed_yeojohnson', 'Bathroom_boxcox', 'Car_yeojohnson', 'Landsize_no_out']

# * Post-Feature Engineering Analysis

# Re-plot histograms, Q-Q plots, and boxplots for the transformed dataset to assess improvements
plot_utils.plot_hist(data=melb_fe, column_list=transformed_quan, rows=2, cols=3)
plot_utils.plot_qq(data=melb_fe, column_list=transformed_quan, rows=2, cols=3)
plot_utils.plot_box(data=melb_fe, column_list=transformed_quan, rows=2, cols=3)

# Assess skewness after transformations
print(melb_fe[transformed_quan].skew())

# Display correlation matrix for the transformed dataset
correlation_matrix_transformed = melb_fe[transformed_quan].corr()
sns.heatmap(correlation_matrix_transformed, annot=True)
plt.title("Correlation Matrix After Transformations")
plt.show()

# Generating pairplot for the transformed variables to visualize distributions, relationships,
# and conduct a basic linearity check (excluding categorical variables)
sns.pairplot(data=melb_fe[transformed_quan])
plt.show()  # Display the pairplot

# Generate descriptive statistics for the transformed dataset
print(melb_fe.describe())

# * Feature Selection and Optimization

# TODO: Pre Feature Selection VIF

# ! Detecting Multicollinearity with VIF (For high condition number)
'''
corr_matrix = X.corr()
sns.heatmap(corr_matrix, annot=True)
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data)  # vif_data shows all VIFs are below the common threshold, so multicollinearity is not a significant concern
'''

# TODO: combining correlated variables into a single composite variable

# TODO: Remove High VIF

# TODO: Feature Selection

# TODO: Post Feature Selection VIF

# * File Exportation

# Construct the full file path
output_file_path_2 = os.path.join(script_dir, '2_transformed_melb_data.csv')

# Export transformed data to a new CSV file
melb_fe.to_csv(output_file_path_2, index=False)