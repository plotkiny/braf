#!/usr/bin/env python3 

# script used to choose optimal hyperameters.
# shouldn't be used by the user, but I include
# it here for completeness

from train_and_eval import main 
from dataclasses import dataclass
from multiprocessing import Pool 
import numpy as np
from tqdm import tqdm
import json 

@dataclass
class Args:
    dataset: str 
    bagging_frac: float = 0.5
    node_depth: int = 4
    output_path: str = '/tmp/hyperparam_opt/'
    K: int = 10
    p: float = 0.5 
    s: int = 100 
    train_frac: float = 0.6 


def runner(args):
    bagging_frac, node_depth = args
    args = Args(dataset='../../diabetes.csv',
                bagging_frac=bagging_frac,
                node_depth=node_depth)
    results = main(args)
    return results['test']['auroc'], results['cv']['auroc']


if __name__ == '__main__':

    args = []
    # for bag_frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    #     for node_depth in range(10):
    for bag_frac in [0.6, 0.7, 0.8, 0.9, 1.0]:
        for node_depth in [3, 5, 8, 10, 15, 20]: 
            args.append((bag_frac, node_depth))

    # args = args[:5]

    pool = Pool(5)
    aurocs = list(tqdm(pool.imap(runner, args), total=len(args)))
    with open(f'/tmp/hyperparam_opt/hyperparam_opt.json', 'w') as fout:
        json.dump(list(zip(args, list(aurocs))), fout)

    # print(f'Best AUROC = {aurocs[i_max]}')
    # print(f'  bagging_frac = {bag_frac}')
    # print(f'  node_depth = {node_depth}')
