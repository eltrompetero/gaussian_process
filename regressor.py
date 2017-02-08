from __future__ import division
import numpy as np
from numba import jit

class GaussianProcessRegressor(object):
    def __init__(self,kernel,beta):
        """
        Params:
        -------
        kernel (function)
            kernel(x,y) that can take two data points and calculate the distance according to the kernel. Must
            be a nopython jit function.
        beta (float)
            variance of noise in observations
        """
        self.kernel = kernel
        self.beta = beta
        self.pred_mean = np.vectorize(self.pred_mean)
        self.pred_var = np.vectorize(self.pred_var) 
        self._define_calc_cov()

        self.cov = None
        self.invCov = None
        self.X = None
        self.Y = None

    def fit(self,X,Y):
        """
        2016-02-07

        Params:
        -------
        X
            n_samples x n_dim. Input.
        Y
            Target variable.
        """
        assert len(X)==len(Y)
        self.X,self.Y = X,Y
        self.cov = self.calc_cov(X)
        self.invCov = np.linalg.inv(self.cov)

    def pred_mean(self,x):
        """
        This is vectorized in __init__().
        """
        k = self.kernel(x,self.X)
        mu = k.T.dot(self.invCov).dot(self.Y)
        return mu
    
    def pred_var(self,x):
        c = self.kernel(x,x) + 1/self.beta
        k = self.kernel(x,self.X)
        return c - k.T.dot(self.invCov).dot(k)

    def _define_calc_cov(self):
        kernel,beta = self.kernel,self.beta

        @jit(nopython=True)
        def calc_cov(X):
            nSamples = len(X)
            cov = np.zeros((nSamples,nSamples))
            for i in xrange(nSamples):
                for j in xrange(i,nSamples):
                    cov[i,j] = kernel(X[i],X[j])
                    if i==j:
                        cov[i,i] += 1/beta
                    else:
                        cov[j,i] = cov[i,j]
            return cov
        self.calc_cov = calc_cov


@jit(nopython=True)
def calc_cov(X,kernel):
    nSamples = len(X)
    cov = np.zeros((nSamples,nSamples))
    for i in xrange(nSamples):
        for j in xrange(i,nSamples):
            cov[i,j] = kernel(X[i],X[j])
            if i==j:
                cov[i,i] += 1/beta
            else:
                cov[j,i] = cov[i,j]
    return cov


