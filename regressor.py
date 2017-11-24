from __future__ import division
import numpy as np
from numba import jit
from scipy.spatial.distance import pdist,squareform

class GaussianProcessRegressor(object):
    def __init__(self,kernel,beta):
        """
        For any d-dimensional input and one dimensional output. From Bishop.

        Parameters
        -------
        kernel (function)
            kernel(x,y) that can take two sets of data points and calculate the distance according to the
            kernel. Must be a nopython jit function.
        beta (float)
            Inverse variance of noise in observations
        """
        self.kernel = kernel
        self.beta = beta
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
            Measured target variable.
        """
        assert len(X)==len(Y)
        self.X,self.Y = np.array(X),np.array(Y)
        self.cov = self.calc_cov(X)
        self.invCov = np.linalg.inv(self.cov)

    def pred_mean(self,x):
        """
        Parameters
        ----------
        x : ndarray
            (n_samples,n_dim)
        """
        assert (not self.X is None) and (not self.Y is None)
        assert x.ndim==2
        k = np.zeros(len(self.X))
        mu = np.zeros(len(x))
        
        for inputIx,ix in enumerate(x):
            for i,x_ in enumerate(self.X):
                k[i] = self.kernel(ix,x_)
            mu[inputIx] = k[None,:].dot(self.invCov).dot(self.Y)
        return mu
    
    def pred_var(self,x):
        """
        Parameters
        ----------
        x : ndarray
            (n_samples,n_dim)
        """
        assert (not self.X is None) and (not self.Y is None)
        assert x.ndim==2
        c = np.zeros(len(x))
        for sampleIx,xi in enumerate(x):
            c[sampleIx] = self.kernel(xi,xi) + 1/self.beta
            k = np.zeros(len(self.X))
            for i,x_ in enumerate(self.X):
                k[i] = self.kernel(xi,x_)
            c[sampleIx] -= k.T.dot(self.invCov).dot(k)
        return c

    def _define_calc_cov(self):
        def calc_cov(X):
            cov = squareform(pdist(X,metric=self.kernel))
            for i,x in enumerate(X):
                cov[i,i] += self.kernel(x,x) + 1/self.beta
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


