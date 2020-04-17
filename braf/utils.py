import pandas as pd 


def mean_impute(df, missing_vals, imputed_vals={}):
    for col, missing_val in missing_vals.items():
        mask = df[col] == missing_val 
        if col not in imputed_vals:
            imputed_vals[col] = df.loc[~mask, col].mean()
        df.loc[mask, col] = imputed_vals[col]

    return imputed_vals


def standardize_features(df, means={}, stds={}, skip={'Outcome'}):
    for col in df.columns:
        if col in skip:
            continue 
        if col not in means:
            means[col], stds[col] = df[col].mean(), df[col].std()
        df[col] = (df[col] - means[col]) / stds[col]

    return means, stds
