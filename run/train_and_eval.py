#!/usr/bin/env python3 

from braf.braf import BRAF
from braf import plot, utils

import pandas as pd 
import numpy as np
from argparse import ArgumentParser 
import os

# set seeds
import random 
random.seed(42)
np.random.seed(42)


def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    df = pd.read_csv(args.dataset)
    plot.df_hist(df, args.output_path, 'raw')

    # mean imputation to fix missing values in the data
    # missing values for these columns are 0 in the dataset
    imputed_val_map = utils.mean_impute(
            df, 
            {k:0 for k in ['BMI', 'BloodPressure', 'Glucose', 'Insulin', 'SkinThickness']}
        )
    plot.df_hist(df, args.output_path, 'imputed')
    
    # standardize the input features to have 0 mean, 1 stdev.
    # the RFs don't care much about this, but this is important 
    # for the KNN, to ensure that Euclidean distance along each
    # dimension is in the same units
    mean_map, std_map = utils.standardize_features(df)

    train_mask = np.random.binomial(n=1, p=args.train_frac, size=df.shape[0]).astype(bool)
    df_train = df[train_mask]
    df_test = df[~train_mask]

    feature_cols = [c for c in df.columns if c != 'Outcome']

    model = BRAF(args.K, args.s, args.p, minority_label=1, 
                 bagging_frac=args.bagging_frac,
                 node_depth=args.node_depth,
                 max_features_per_node=3, # ~sqrt(8 features)
                 n_search_pts=20)
    model.fit(df_train[feature_cols].values, df_train['Outcome'].values)

    y_test = df_test['Outcome'].values
    yhat_test = model.predict(df_test[feature_cols].values)
    plot.prediction_hist(yhat_test, y_test, args.output_path)

    auroc = plot.roc(yhat_test, y_test, args.output_path)
    auprc, precision, recall = plot.prc(yhat_test, y_test, args.output_path)

    # print(rf)
    return
    


if __name__ == '__main__':
    parser = ArgumentParser('Trainer')
    parser.add_argument('--dataset', help='path to diabetes.csv', required=True) 
    parser.add_argument('--output_path', required=True)
    parser.add_argument('-K', type=int, default=10, help='number of neighbors for critical area')
    parser.add_argument('-p', type=float, default=0.5, help='fraction of trees applied to critical area')
    parser.add_argument('-s', type=int, default=100, help='total number of trees to train')
    parser.add_argument('--train_frac', type=float, default=0.6)
    parser.add_argument('--bagging_frac', type=float, default=0.6)
    parser.add_argument('--node_depth', type=int, default=4)
    args = parser.parse_args()

    main(args)
