import numpy as np 
import matplotlib.pyplot as plt 

EPS = 1e-12

def _plot_to(path):
    plt.tight_layout()
    print(f'Saving to {path}')
    for ext in ('pdf', 'png'):
        plt.savefig(f'{path}.{ext}')
    plt.clf()


def df_hist(df, path, suffix):
    df.hist(bins=100)
    _plot_to(f'{path}/features_{suffix}')


def prediction_hist(yhat, y, path):
    bins = np.linspace(0, 1, 25)
    plt.hist(yhat[y==0], bins=bins, alpha=0.5, label='Outcome 0')
    plt.hist(yhat[y==1], bins=bins, alpha=0.5, label='Outcome 1')
    plt.legend()
    _plot_to(f'{path}/yhat')


def _fp(yhat, y, c):
    mask = yhat > c 
    fps = np.logical_and(mask, y==0).sum()
    total = (y==0).sum() + EPS
    return fps / total 


def _tp(yhat, y, c):
    mask = yhat > c 
    tps = np.logical_and(mask, y==1).sum()
    total = (y==1).sum() + EPS
    return tps / total 


def roc(yhat, y, path):
    thresholds = np.linspace(1, 0, 25) 
    fp = [_fp(yhat, y, t) for t in thresholds] 
    tp = [_tp(yhat, y, t) for t in thresholds] 

    auc = np.trapz(tp, x=fp)

    plt.plot(fp, tp, label=f'AUC={auc:.3f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    _plot_to(f'{path}/roc')

    return auc


def _prec(yhat, y, c):
    mask = yhat > c
    tps = np.logical_and(mask, y==1).sum()
    total = mask.sum() + EPS
    return tps / total 
    

def prc(yhat, y, path):
    thresholds = np.linspace(1, 0, 25) 
    prec = [_prec(yhat, y, t) for t in thresholds] 
    tp = [_tp(yhat, y, t) for t in thresholds] 

    auc = np.trapz(prec, x=tp)

    plt.plot(tp, prec, label=f'AUC={auc:.3f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    _plot_to(f'{path}/prc')

    return auc, _prec(yhat, y, 0.5), _tp(yhat, y, 0.5)
