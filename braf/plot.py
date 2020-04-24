import numpy as np 
import matplotlib.pyplot as plt 

EPS = 1e-12


def _plot_to(path):
    '''save a plot'''
    plt.tight_layout()
    # print(f'Saving to {path}')
    for ext in ('pdf', 'png'):
        plt.savefig(f'{path}.{ext}')
    plt.clf()


def df_hist(df, path, suffix):
    '''plot all columns as a dataframe as histograms'''
    df.hist(bins=100)
    _plot_to(f'{path}/features_{suffix}')


def prediction_hist(yhat, y, path):
    '''plot the prediction scores, assuming two classes'''
    bins = np.linspace(0, 1, 25)
    plt.hist(yhat[y==0], bins=bins, alpha=0.5, label='Outcome 0')
    plt.hist(yhat[y==1], bins=bins, alpha=0.5, label='Outcome 1')
    plt.legend()
    _plot_to(f'{path}/yhat')


def _fp(yhat, y, c):
    '''compute false positives with a decision threshold c'''
    mask = yhat > c 
    fps = np.logical_and(mask, y==0).sum()
    total = (y==0).sum() + EPS
    return fps / total 


def _tp(yhat, y, c):
    '''compute true positives with a decision threshold c'''
    mask = yhat > c 
    tps = np.logical_and(mask, y==1).sum()
    total = (y==1).sum() + EPS
    return tps / total 


def roc(yhat, y, path, suffix):
    '''
    compute ROC curve and return AUC

    Parameters:

    - yhat (ndarray): prediction score for the positive class
    - y (ndarray): binary label
    - path (str): output directory to which the plot should be saved 
    - suffix (str): label to make the plot filename unique

    Returns:
    
    - float: area under the ROC curve
    '''

    thresholds = np.linspace(1, 0, 25) 
    fp = [_fp(yhat, y, t) for t in thresholds] 
    tp = [_tp(yhat, y, t) for t in thresholds] 

    auc = np.trapz(tp, x=fp)

    plt.plot(fp, tp, label=f'AUC={auc:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    _plot_to(f'{path}/roc_{suffix}')

    return auc


def _prec(yhat, y, c):
    '''compute precision with a decision threshold c'''
    mask = yhat > c
    tps = np.logical_and(mask, y==1).sum()
    total = mask.sum() + EPS
    return tps / total 
    

def prc(yhat, y, path, suffix):
    '''
    compute PRC curve and return AUC

    Parameters:

    - yhat (ndarray): prediction score for the positive class
    - y (ndarray): binary label
    - path (str): output directory to which the plot should be saved 
    - suffix (str): label to make the plot filename unique

    Returns:
    
    - float: area under the PRC curve
    '''

    thresholds = np.linspace(1, 0, 25) 
    prec = [_prec(yhat, y, t) for t in thresholds] 
    tp = [_tp(yhat, y, t) for t in thresholds] 

    auc = np.trapz(prec, x=tp)

    plt.plot(tp, prec, label=f'AUC={auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    _plot_to(f'{path}/prc_{suffix}')

    return auc, _prec(yhat, y, 0.5), _tp(yhat, y, 0.5)
