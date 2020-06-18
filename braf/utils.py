#!/usr/bin/env python3

import pandas as pd
from braf.knn import KNN


def col_impute(df, missing_vals, mthd='median', **kwargs):
    '''
    choose impute method

    Parameters:

    - df (Dataframe): dataframe
    - missing_vals (dict): a dictionary mapping col_name -> value
                         where value indicates the feature is missing
    -mthd: impute method: median, mean or knn

    Returns:

    - dict: a completed copy of imputed_vals

    '''

    mthd_types = ['mean', 'median', 'knn']
    mthd = mthd.lower()

    if mthd not in mthd_types:
        raise ValueError("Imputation must be one of the following {}".format(mthd_types))

    imputed_val_map = None

    if mthd in ['median', 'mean']:
        imputed_val_map = stat_impute(df, missing_vals, mthd)
    elif mthd == 'knn':
        imputed_val_map = knn_impute(df, missing_vals, **kwargs)

    return imputed_val_map


def knn_impute(df, missing_vals, imputed_vals={}, **kwargs):
    '''
    impute missing values in-place using knn

    Parameters:

    - df (Dataframe): dataframe
    - missing_vals (dict): a dictionary mapping col_name -> value
                         where value indicates the feature is missing
    -kwargs (dict): parameters for performing nearest neighbor clustering

    Returns:

    - dict: a completed copy of imputed_vals

    '''

    feature_cols = kwargs['feature']
    k = kwargs['K']
    metric = kwargs['metric']
    min_label = kwargs['minority_label']
    y = kwargs['y']

    for col, missing_val in missing_vals.items():
        mask = df[col] == missing_val
        mask_index = mask[mask == True].index.tolist()
        if col not in imputed_vals:
            df_unmask = df[mask][feature_cols].values
            df_mask = df[~mask]
            df_mask = df_mask[feature_cols].values
            #df_mask = df_mask[df_mask[y] != min_label][feature_cols].values
            critical_nghbrs = KNN.get_neighbors(df_unmask, df_mask, k=k, metric=metric)
            critical_nghbrs_tup = list(zip(mask_index, critical_nghbrs))
            for indx, nghbrs in critical_nghbrs_tup:
                stat = df[col].iloc[nghbrs].median()  # using median here
                stat = round(stat, 1)
                imputed_vals[col] = stat
                df.loc[indx, col] = imputed_vals[col]

    return imputed_vals


def stat_impute(df, missing_vals, mthd, imputed_vals={}):
    '''
    impute missing values in-place using sufficient statistics

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
            df_mask = df.loc[~mask, col]
            if mthd == 'median':
                stat = df_mask.median()
            elif mthd == 'mean':
                stat = df_mask.mean()
            stat = round(stat, 1)
            imputed_vals[col] = stat
        df.loc[mask, col] = imputed_vals[col]

    return imputed_vals


def standardize_normal(df, col, means, stds, mask=None):
    if isinstance(mask, pd.Series):
        means[col], stds[col] = df[~mask][col].mean(), df[~mask][col].std()
        df.loc[df[~mask].index.tolist(), col] = \
            (df.loc[df[~mask].index.tolist(), col] - means[col]) / stds[col]
    else:
        means[col], stds[col] = df[col].mean(), df[col].std()
        df.loc[:, col] = (df[col] - means[col]) / stds[col]

    return means, stds


def standardize_min_max(df, col, means, stds, mask=None):
    if isinstance(mask, pd.Series):
        X = df[~mask][col]
        means[col], stds[col] = X.mean(), X.std()
        df.loc[df[~mask].index.tolist(), col] = (X - X.min()) / (X.max() - X.min())
    else:
        X = df[col]
        means[col], stds[col] = X.mean(), X.std()
        df.loc[:, col] = (X - X.min()) / (X.max() - X.min())

    return means, stds


def standardize_features(df, missing_vals, scaler='normal', means={}, stds={}, skip={'Outcome'}):
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

    if scaler == "min_max":
        func = standardize_min_max
    elif scaler == "normal":
        func = standardize_normal

    for col in df.columns:
        if col in skip:
            continue
        if col not in means:
            if col in missing_vals:
                missing_val = missing_vals[col]
                mask = df[col] == missing_val
                means, std = func(df, col, means, stds, mask=mask)
            else:
                means, std = func(df, col, means, stds)

    return means, stds
