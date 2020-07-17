#!/usr/bin/env python3 

from braf.braf import BRAF
from braf import plot, utils

import pandas as pd
import numpy as np
from argparse import ArgumentParser
import os
import pickle

# set seeds for reproducibility
import random

random.seed(42)
np.random.seed(42)


def main(args):
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    df = pd.read_csv(args.dataset)
    plot.df_hist(df, args.output_path, 'raw')

    # we consider anything besides 'Outcome' as a predictive feature
    feature_cols = [c for c in df.columns if c != 'Outcome']

    kwargs = {
        'feature': feature_cols,
        'K': 30,
        'metric': 'cosine',
        'minority_label': 1,
        'y': 'Outcome'
    }

    # mean imputation to fix missing values in the data.
    # missing values for these columns are 0 in the dataset
    imputed_val_map = utils.col_impute(
        df,
        {k: 0 for k in ['BMI', 'BloodPressure', 'Glucose', 'Insulin', 'SkinThickness']},
        mthd='knn',
        **kwargs
    )
    plot.df_hist(df, args.output_path, 'imputed')

    missing_vals = {k: 0 for k in ['BMI', 'BloodPressure', 'Glucose', 'Insulin', 'SkinThickness']}

    # standardize the input features to have 0 mean, 1 stdev.
    # the RFs don't care much about this, but this is important
    # for the KNN, to ensure that Euclidean distance along each
    # dimension is in the same units
    mean_map, std_map = utils.standardize_features(df, missing_vals, scaler='min_max')

    plot.df_hist(df, args.output_path, 'standardized')

    # split the dataset in train and test sets
    train_mask = np.random.binomial(n=1, p=args.train_frac, size=df.shape[0]).astype(bool)
    df_train = df[train_mask]
    df_test = df[~train_mask]

    cv_yhats = []
    cv_ys = []
    cv_models = []
    n_train = df_train.shape[0]

    for i_cv in range(10):
        # 10-fold cross validation
        test_range = (i_cv * n_train // 10, (i_cv + 1) * n_train // 10)

        # split the dataset into 10 segments and take the i_cv-segment
        # as the cross val set for this fold. rest is train.
        idcs = np.arange(n_train)
        cv_test_mask = np.logical_and(idcs > test_range[0], idcs <= test_range[1])
        df_train_cv = df_train[~cv_test_mask]
        df_test_cv = df_train[cv_test_mask]

        # train BRAF on the train subset
        model = BRAF(args.K, args.s, args.p, minority_label=1,
                     bagging_frac=args.bagging_frac,
                     node_depth=args.node_depth,
                     max_features_per_node=3,  # ~sqrt(8 features)
                     n_search_pts=20)
        model.fit(df_train_cv[feature_cols].values, df_train_cv['Outcome'].values)

        # run inference on the test subset
        yhat = model.predict(df_test_cv[feature_cols].values)

        # save the predictions for this fold
        cv_yhats.append(yhat)
        cv_ys.append(df_test_cv['Outcome'].values)

        # save the model for this fold
        cv_models.append(model)

    # concatenate the cross-val fold predictions/labels into a single array
    # for easy plotting
    yhat_cv = np.concatenate(cv_yhats, axis=0)
    y_cv = np.concatenate(cv_ys, axis=0)

    # make the plots + get summary stats
    auroc_cv = plot.roc(yhat_cv, y_cv, args.output_path, 'cv')
    auprc_cv, precision_cv, recall_cv = plot.prc(yhat_cv, y_cv, args.output_path, 'cv')

    # run the 10 models and take the average of the
    # ensemble as the prediction
    y_test = df_test['Outcome'].values
    test_yhats = []
    for model in cv_models:
        test_yhats.append(model.predict(df_test[feature_cols].values))
    yhat_test = np.mean(test_yhats, axis=0)

    # make the plots + get summary stats
    auroc_test = plot.roc(yhat_test, y_test, args.output_path, 'test')
    auprc_test, precision_test, recall_test = plot.prc(yhat_test, y_test, args.output_path, 'test')

    results_cv = {
        'auroc': auroc_cv,
        'auprc': auprc_cv,
        'precision': precision_cv,
        'recall': recall_cv
    }

    results_test = {
        'auroc': auroc_test,
        'auprc': auprc_test,
        'precision': precision_test,
        'recall': recall_test
    }

    results = {'cv': results_cv, 'test': results_test}

    # save all these things to disk, as they're needed to run
    # inference (except results and args):
    #  - results, args: so we know what was run
    #  - mean_map, std_map: used to standardize any new data
    #  - imputed_val_map: used to impute missing values in any new data
    #  - models: list of the models trained on each cross-val fold
    artifacts = {
        'args': args,
        'results': results,
        'mean_map': mean_map,
        'std_map': std_map,
        'imputed_val_map': imputed_val_map,
        'models': cv_models
    }
    with open(f'{args.output_path}/model_artifacts.pkl', 'wb') as fpkl:
        pickle.dump(artifacts, fpkl)

    return results


if __name__ == '__main__':
    parser = ArgumentParser('Trainer')
    parser.add_argument('--dataset', help='path to diabetes.csv', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('-K', type=int, default=10, help='number of neighbors for critical area')
    parser.add_argument('-p', type=float, default=0.5, help='fraction of trees applied to critical area')
    parser.add_argument('-s', type=int, default=100, help='total number of trees to train')
    parser.add_argument('--train_frac', type=float, default=0.8, help='fraction of the dataset used as a training set')

    # the following two defaults are optimal for K=10, p=0.5, s=100
    parser.add_argument('--bagging_frac', type=float, default=0.8,
                        help='fraction of the training set sampled for each bag')
    parser.add_argument('--node_depth', type=int, default=3, help='maximum depth of each decision tree')
    args = parser.parse_args()

    results = main(args)

    print(f'ROC and PRC curves placed in {args.output_path}')
    print(f'Cross-validation set results:')
    for k in ('auroc', 'auprc', 'precision', 'recall'):
        print(f'   {k}={results["cv"][k]:.3f}')
    print(f'Test set results:')
    for k in ('auroc', 'auprc', 'precision', 'recall'):
        print(f'   {k}={results["test"][k]:.3f}')
