# =============================================================================================== #
# Module for Gaussian process regression.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# =============================================================================================== #

from __future__ import division
import numpy as np
from numba import jit
from scipy.spatial.distance import pdist,squareform
from itertools import combinations



class GaussianProcessRegressor(object):
    def __init__(self,kernel,beta):
        """
        For any d-dimensional input and one dimensional output. From Bishop.

        Parameters
        ----------
        kernel : function
            kernel(x,y) that can take two sets of data points and calculate the distance according to the
            kernel. Must be a nopython jit function.
        beta : float
            Inverse variance of noise in observations. In other words, 1/beta is added to the
            diagonal entries of the covariance matrix.
        """
        assert beta>0

        self.kernel = kernel
        self.beta = beta
        self._define_calc_cov()

        self.cov = None
        self.invCov = None
        self.X = None
        self.Y = None

    def fit(self,X,Y):
        """
        Parameters
        ----------
        X : ndarray
            n_samples x n_dim. Input.
        Y : ndarray
            Measured target variable.
        """
        assert len(X)==len(Y)

        self.X,self.Y = np.array(X),np.array(Y)
        self.cov = self.calc_cov(X)
        self.invCov = np.linalg.inv(self.cov)

    def predict(self,x,X=None,Y=None,inv_cov=None,return_std=False,verbose=True):
        """
        Parameters
        ----------
        x : ndarray
            Dim (n_samples,n_dim).
        X : ndarray
        Y : ndarray
        inv_cov : ndarray
        return_std : bool,False

        Returns
        -------
        mu : ndarray
        err : ndarray
        """
        # Check args.
        if X is None:
            X=self.X
            assert (not X is None) 
        if Y is None:
            Y=self.Y
            assert (not Y is None)
        assert x.ndim==2
        if inv_cov is None:
            inv_cov=self.invCov
        else:
            assert inv_cov.shape[0]==inv_cov.shape[1]
        k = np.zeros(len(X))
        mu = np.zeros(len(x))
        
        # Case where only predicted means are returned.
        if not return_std:
            for sampleIx,xi in enumerate(x):
                for i,x_ in enumerate(X):
                    k[i] = self.kernel(xi,x_)
                mu[sampleIx] = k[None,:].dot(inv_cov).dot(Y)
            return mu
        
        # Case where both means and errors are returned.
        c = np.zeros(len(x))
        for sampleIx,xi in enumerate(x):
            c[sampleIx] = self.kernel(xi,xi) + 1/self.beta
            k = np.zeros(len(X))
            for i,x_ in enumerate(X):
                k[i] = self.kernel(xi,x_)
            mu[sampleIx] = k[None,:].dot(inv_cov).dot(Y)
            c[sampleIx] -= k.T.dot(inv_cov).dot(k)
        
        # For debugging purposes. In case some of these values are very small and potentially
        # stemming from precision and not a bad kernel.
        if (c<0).any() and verbose:
            print "Estimated errors are not all positive. Printing negative entries:"
            print c[c<0]
        return mu,np.sqrt(c)

    def leave_one_out_predict(self,x,i,return_std=False):
        """Prediction at x when ith data point from training data is left out. This is simply calculated
        by leaving the ith col and row out from the covariance matrix.

        Parameters
        ----------
        x : ndarray
        i : int
            Index of training data point to leave out.
        return_std : bool,False

        Returns
        -------
        y : ndarray
            For each row of x, a vector of leave one out predictions is returns such that y is of
            dim (n_samples,n_training_sample).
        """
        n=self.cov.shape[0]
        X=self.X[np.delete(range(n),i,axis=0)]
        Y=self.Y[np.delete(range(n),i,axis=0)]
        invCov=np.linalg.inv( self.cov[np.delete(range(n),i),:][:,np.delete(range(n),i)] )

        return self.predict(x,X=X,Y=Y,inv_cov=invCov,return_std=return_std)

    def ocv_error(self):
        """Calculate ordinary cross validation error on point x with measured function value y.
        https://en.wikipedia.org/wiki/Projection_matrix.

        Parameters
        ----------
        x : ndarray
        y : ndarray

        Returns
        -------
        ocv : float
        """
        inv=np.linalg.inv
        nSample=len(self.X)
        hatMatrix=self.X.dot(inv(self.X.T.dot(self.X))).dot(self.X.T)
        ypred=self.predict(self.X)
        return np.mean(( (self.Y-ypred)/(1-hatMatrix[np.diag_indices(nSample)]) )**2)

    def gcv_error(self):
        """Calculate generalized cross validation error on point x with measured function value y.

        Parameters
        ----------
        x : ndarray
        y : ndarray

        Returns
        -------
        gcv : float
        """
        inv=np.linalg.inv
        nSample=len(self.X)
        hatMatrix=self.X.dot(inv(self.X.T.dot(self.X))).dot(self.X.T)
        ypred=self.predict(self.X)
        return np.mean( (self.Y-ypred)**2 )/(1-np.trace(hatMatrix)/nSample)**2

    def _define_calc_cov(self):
        """For calculating covariance matrix. Could be sped up by jitting everything."""
        #@jit(nopython=True)
        def calc_cov(X,kernel=self.kernel):
            nSamples = len(X)
            cov = np.zeros((nSamples,nSamples))
            for i,j in combinations(range(nSamples),2):
                cov[i,j] = cov[j,i] = kernel(X[i],X[j])
            for i in xrange(nSamples):
                cov[i,i] += 1/self.beta
            return cov

        self.calc_cov = calc_cov
    
    def log_likelihood(self):
        """Log-likelihood of the current state of the GPR.
        """
        det=np.linalg.slogdet(self.cov)
        assert det[0]>0,(det,self.cov.min())
        return ( -.5*det[1]
                 -.5*self.Y.dot(np.linalg.inv(self.cov)).dot(self.Y)
                 -len(self.X)/2*np.log(2*np.pi) )

    def run_checks(self):
        """Run some basic checks to make sure GPR was fit okay.
        """
        assert not self.X is None,"Must train GPR on some data first."

        # Check that covariance matrix is well conditioned.
        det=np.linalg.slogdet(self.cov)
        if det[0]==-1:
            print "ERR: Determinant is negative."
        elif det[1]<-10:
            print "ERR: Covariance matrix is ill-conditioned."
        else:
            print "OK: Covariance matrix is okay."

        # Check that errors are positive.
        mu,std=self.predict(self.X,return_std=True)
        if (std<0).any():
            print "ERR: Some errors are negative."
        else:
            print "OK: All errors are positive."
#end GaussianProcessRegressor


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

def define_rbf(el):
    """A simple rbf kernel with length scale."""
    @jit(nopython=True)
    def rbf(x,y):
        d=0
        for i in xrange(len(x)):
            d+=(x[i]-y[i])**2
        return np.exp(-np.sqrt(d)/2/el**2)
    return rbf
