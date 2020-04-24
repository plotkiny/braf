## BRAF

The `braf` module implements the Biased Random Forest ([link](https://researchportal.port.ac.uk/portal/files/11841186/Biased_Random_Forest_v08_Accepted_Bader.pdf)). The only dependencies are
`python3.7+`, `numpy`, `pandas`, and `matplotlib`. To install, just run:

```
$ python setup.py install
```

The script `run/train_and_evaluate.py` will train on a supplied dataset. Plots, model 
artifacts, and results will be saved to a supplied output directory. Some results
are also printed to stdout. Example usage:

```
$ ./train_and_eval.py --dataset ../../diabetes.csv --output_path ../../results/
ROC and PRC curves placed in ../../results/
Cross-validation set results:
   auroc=0.836
   auprc=0.686
   precision=0.667
   recall=0.590
Test set results:
   auroc=0.814
   auprc=0.682
   precision=0.628
   recall=0.584
```


Also provided is a helper script `run/hyperparam_opt.py`, which will train and evaluate over
a grid of hyperparameter options. This was used to set defaults in `train_and_evaluate.py`,
aside from the provided defaults (K=10, s=100, p=0.5).
