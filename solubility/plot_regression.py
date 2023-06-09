import pandas as pd
import numpy as np
import seaborn as sns
import bootstrapped.bootstrap as bs
from scipy.stats import pearsonr

### The bootstrapping routines were adapted from https://github.com/openforcefield/protein-ligand-benchmark-livecoms

def MAE_function(values):
    return np.mean(values,axis=1)

def pearsonr_function(a):
  s = a.shape
  tmp = np.array([])
  if len(s) == 2:
    tmp = pearsonr(a[:,0],a[:,1])
  else:
    for i in range(s[0]):
      x = pearsonr(a[i,:,0],a[i,:,1])[0]
      if x == np.nan:
        tmp = np.append(tmp,0.0)
      else:
        tmp = np.append(tmp,x)        
  return tmp

def bootstrap_MAE(truth, pred):
    error = np.abs(truth-pred)
    bs_dist = bs.bootstrap(np.reshape(error.values,(len(error),-1)),
             stat_func=MAE_function, alpha=0.5, is_pivotal=True, return_distribution=True)
    return np.percentile(bs_dist,2.5),np.mean(error),np.percentile(bs_dist,97.5)

def bootstrap_pearsonr(truth, pred):
    a = np.array(list(zip(truth,pred)))
    bs_dist = bs.bootstrap(a,
             stat_func=pearsonr_function, alpha=0.5, is_pivotal=True, return_distribution=True)
    return np.percentile(bs_dist,2.5),pearsonr(truth,pred)[0],np.percentile(bs_dist,97.5)

def plot_regression(truth, pred):
    tmp_df = pd.DataFrame({"truth": truth, "pred": pred})
    mae_lb, mae, mae_ub = bootstrap_MAE(truth, pred)
    r_lb, r, r_ub = bootstrap_pearsonr(truth, pred)
    sns.set_context("notebook")
    g = sns.lmplot(x='truth',y='pred',data=tmp_df,
            scatter_kws=dict(alpha=0.2, s=20, color='blue', edgecolors='white'))
    g.axes[0][0].set_xlabel("Experimental LogS")
    g.axes[0][0].set_ylabel("Predicted LogS")
    g.axes[0][0].set_ylim(-10,0)
    g.axes[0][0].text(-6.5,-0.5,f"r={r:.2f} [{r_lb:.2f},{r_ub:.2f}]")
    g.axes[0][0].text(-6.5,-1.5,f"MAE={mae:.2f} [{mae_lb:.2f},{mae_ub:.2f}]")

