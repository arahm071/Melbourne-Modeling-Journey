import pandas as pd

def concat_dummies(df, cat_list):
    """
    Generate dummy variables for specified categorical columns and concatenate
    them to the original DataFrame.
    
    This function takes a DataFrame and a list of column names that are 
    categorical. For each categorical column, it creates dummy variables 
    (excluding the first category to avoid multicollinearity) and appends 
    these as new columns to a copy of the original DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The original DataFrame.
    - cat_list (list of str): A list of column names in `df` that are 
      categorical and for which dummy variables will be created.
    
    Returns:
    - pd.DataFrame: A copy of `df` with added dummy variables for the specified 
      categorical columns.
    """
    dummy_df = df.copy()

    for cat in cat_list:
        dummy_var = pd.get_dummies(df[cat], drop_first=True, dtype=int)
        
        # Concatenate the dummy variables to the copy of the DataFrame
        dummy_df = pd.concat([dummy_df, dummy_var], axis=1)

    return dummy_df
