import pandas as pd 


def mean_impute(df, missing_vals, imputed_vals={}):
    '''
    impute missing values in-place

    Parameters:

    - df (Dataframe): dataframe 
    - missing_vals (dict): a dictionary mapping col_name -> value 
                         where value indicates the feature is missing
    - imputed_vals (dict): an optional dictionary mapping col_name -> mean_value 
                         where mean_value is the mean of non-missing values.
                         if not provided, it will be computed from df. 
    
    Returns:
    
    - dict: a completed copy of imputed_vals 
    '''

    for col, missing_val in missing_vals.items():
        mask = df[col] == missing_val 
        if col not in imputed_vals:
            imputed_vals[col] = df.loc[~mask, col].mean()
        df.loc[mask, col] = imputed_vals[col]

    return imputed_vals


def standardize_features(df, means={}, stds={}, skip={'Outcome'}):
    '''
    standardize features in-place

    Parameters:
    
    - df (Dataframe): dataframe 
    - means (dict): a dictionary mapping col_name -> mean value. if not provided,
                  it will be computed from df.
    - stds (dict):  a dictionary mapping col_name -> std value. if not provided,
                  it will be computed from df.
    - skip (set): a set of columns to skip, e.g. label columns
    
    Returns:
    
    - dict: a completed copy of means 
    - dict: a completed copy of stds 
    '''

    for col in df.columns:
        if col in skip:
            continue 
        if col not in means:
            means[col], stds[col] = df[col].mean(), df[col].std()
        df[col] = (df[col] - means[col]) / stds[col]

    return means, stds
