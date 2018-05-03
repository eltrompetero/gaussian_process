# Data preprocessing.
from __future__ import division
from misc.utils import unique_rows
import numpy as np


def preprocess_average_repeat_values(X,Y):
    """For any repeat data points X, take the average of the measured values Y.
    
    Parameters
    ----------
    X : ndarray
    Y : ndarray
    
    Returns
    -------
    XnoRepeats : ndarray
    YnoRepeats : ndarray
    """
    Xsquished=X[unique_rows(X)]
    Ysquished=np.zeros(len(Xsquished))
    
    for i,row in enumerate(Xsquished):
        Ysquished[i]=Y[(row[None,:]==X).all(1)].mean()
        
    return Xsquished,Ysquished
