import matplotlib.pyplot as plt  # Plotting library for creating static, interactive visualizations
import seaborn as sns            # Data visualization library based on matplotlib
from scipy import stats          # Library for scientific and technical computing

def plot_hist(data, column_list, rows, cols, fig_x=15, fig_y=15):
    """
    Plots histograms for each specified column in a dataset to analyze data distribution and skewness.

    This function creates a grid of histograms for visualizing the distribution of values within each specified 
    column. This can help in identifying the skewness of the data distribution and in analyzing the presence of 
    outliers or unusual data points.

    Parameters:
    - data (DataFrame): The dataset containing the data to plot.
    - column_list (list of str): List of column names for which histograms will be created.
    - rows (int): Number of rows in the subplot grid.
    - cols (int): Number of columns in the subplot grid.
    - fig_x (int, optional): Width of the figure in inches. Default is 15.
    - fig_y (int, optional): Height of the figure in inches. Default is 15.

    Returns:
    None
    """
    # Initialize the subplot grid with specified dimensions
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_x, fig_y))
    row_count, col_count = 0, 0  # Track the current row and column indices

    # Iterate through each column name, plotting a histogram for each
    for column_name in column_list:
        sns.histplot(data=data, x=column_name, ax=axs[row_count, col_count])
        # Set the title for the current plot
        axs[row_count, col_count].set_title(f'Distribution of {column_name}')
        col_count += 1  # Move to the next column position

        # Check if the current row is filled; if so, move to the next row
        if col_count >= cols:
            col_count = 0  # Reset column index for new row
            row_count += 1  # Move to the next row

    plt.tight_layout()  # Adjust subplot parameters to prevent overlap of plot elements
    plt.show()  # Display the grid of histograms

    
def plot_qq(data, column_list, rows, cols, fig_x=15, fig_y=15):
    """
    Plots a grid of Q-Q (quantile-quantile) plots for the specified columns in the dataset.

    This function is used to assess if the distribution of data in the specified columns follows a normal distribution
    by comparing their quantiles to the quantiles of a standard normal distribution. Each plot is placed within a 
    grid defined by the number of rows and columns.

    Parameters:
    - data (DataFrame): The DataFrame containing the data to plot.
    - column_list (list of str): List of column names for which Q-Q plots will be created.
    - rows (int): Number of rows in the subplot grid.
    - cols (int): Number of columns in the subplot grid.
    - fig_x (int, optional): Width of the figure in inches. Default is 15.
    - fig_y (int, optional): Height of the figure in inches. Default is 15.

    Returns:
    None
    """
    # Initialize the subplot grid with specified dimensions and adjust spacing
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_x, fig_y))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)  # Adjust the spacing between plots for clarity

    # Flatten the array of axes for straightforward iteration
    axs = axs.flatten()

    # Iterate through each column name, generating a Q-Q plot for each
    for i, column_name in enumerate(column_list):
        # Create Q-Q plot for the current column against a normal distribution
        stats.probplot(data[column_name], dist="norm", plot=axs[i])
        # Set the title for the current plot
        axs[i].set_title(f'Q-Q Plot of {column_name}')

    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    plt.show()  # Display the grid of Q-Q plots


def plot_box(data, column_list, rows, cols, price=False, fig_x=15, fig_y=15):
    """
    Plots box plots for each specified column in a dataset, optionally against a 'Price' column, to identify outliers.

    This function creates a grid of box plots for each column specified in the column_list. If the price parameter
    is set to True, each box plot will compare the values of the column against the 'Price' column. This is useful
    for visualizing the spread of data and identifying outliers within the dataset.

    Parameters:
    - data (DataFrame): The dataset containing the data to plot.
    - column_list (list of str): List of column names for which box plots will be created.
    - rows (int): Number of rows in the subplot grid.
    - cols (int): Number of columns in the subplot grid.
    - price (bool, optional): Determines if box plots should be plotted against a 'Price' column. Default is False.
    - fig_x (int, optional): Width of the figure in inches. Default is 15.
    - fig_y (int, optional): Height of the figure in inches. Default is 15.

    Returns:
    None
    """
    # Initialize the subplot grid with specified dimensions
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_x, fig_y))
    row_count, col_count = 0, 0  # Track the current row and column indices

    if price:
        # Plot each specified column against 'Price' if price is True
        for column_name in column_list:
            sns.boxplot(data=data, x=column_name, y='Price', ax=axs[row_count, col_count])
            axs[row_count, col_count].set_title(f'Boxplot of Price vs. {column_name}')
            col_count += 1  # Increment column index

            # Move to the next row if the current row is filled
            if col_count >= cols:
                col_count = 0  # Reset column index for new row
                row_count += 1  # Move to the next row
    else:
        # Plot box plots for each specified column without comparing to 'Price'
        for column_name in column_list:
            sns.boxplot(data=data, x=column_name, ax=axs[row_count, col_count])
            axs[row_count, col_count].set_title(f'Boxplot of {column_name}')
            col_count += 1  # Increment column index

            # Check if the current row is filled and move to the next row if so
            if col_count >= cols:
                col_count = 0  # Reset column index for new row
                row_count += 1  # Move to the next row

    plt.tight_layout()  # Adjust layout to prevent overlap of plot elements
    plt.show()  # Display the grid of box plots


def plot_violin(data, column_list, rows, cols, fig_x=15, fig_y=15):
    """
    Plots violin plots for each specified column in a dataset against a 'Price' column to identify data distribution.

    The function creates a grid of violin plots where each plot represents the distribution of data in one of the 
    specified columns against the 'Price' column, allowing for the visualization of data spread and potential outliers.

    Parameters:
    - data (DataFrame): The dataset containing the data to plot.
    - column_list (list of str): List of column names for which violin plots will be created.
    - rows (int): Number of rows in the subplot grid.
    - cols (int): Number of columns in the subplot grid.
    - fig_x (int, optional): Width of the figure in inches. Default is 15.
    - fig_y (int, optional): Height of the figure in inches. Default is 15.

    Returns:
    None
    """
    # Initialize the subplot grid with specified dimensions
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(fig_x, fig_y))
    row_count, col_count = 0, 0  # Track the current row and column indices

    # Iterate through the list of columns, creating a violin plot for each
    for column_name in column_list:
        # Plot violin plot for the current column against 'Price'
        sns.violinplot(data=data, x=column_name, y='Price', ax=axs[row_count, col_count])
        # Set the title of the current plot
        axs[row_count, col_count].set_title(f'Violin Plot of Price vs. {column_name}')
        
        col_count += 1  # Move to the next column in the grid

        # Check if the current row is filled; move to the next row if so
        if col_count >= cols:
            col_count = 0  # Reset column index
            row_count += 1  # Move to the next row

    plt.tight_layout()  # Adjust subplot parameters to give specified padding
    plt.show()  # Display the completed grid of violin plots