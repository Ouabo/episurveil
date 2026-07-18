import numpy as np
def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y)-p)**2)))
def mae(y,p): return float(np.mean(np.abs(np.asarray(y)-p)))
def coverage(y,lo,hi): return float(np.mean((np.asarray(y)>=lo)&(np.asarray(y)<=hi)))
