# =============================================================================================== #
# Module for Gaussian process regression.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# =============================================================================================== #

from __future__ import division
import numpy as np
from numba import jit
from scipy.spatial.distance import pdist,squareform
from itertools import combinations
from scipy.optimize import minimize



class BlockGPR(object):
    def __init__(self,n,dist=None):
        self.n=n  # no. of disjoint blocks
        self.dist=dist or lambda x,y:np.linalg.norm(x-y)
        
        # Default block specific parameters.
        self.noisei=np.zeros(n)+.5
        self.mui=np.zeros(n)
        self.lengthi=np.zeros(n)+1
        self.coeffi=np.ones(n)
        self.smoothnessi=np.zeros(n)+.5

        # Default parameters on common landscape shared by all block kernels.
        self.noiseAll=.5
        self.muAll=0.
        self.lengthAll=1
        self.coeffAll=1
        self.smoothnessAll=.5
        
        self._setup_kernels()
        self._define_kernel()
        
        self._update_gp()
        
    def _update_gp(self):
        self.gp=GaussianProcessRegressor(self.kernel,np.inf)
    
    def _setup_kernels(self):
        """First time instance is declared."""
        self.blockKernel=[]
        for el,coef,nu in zip(self.lengthi,self.coeffi,self.smoothnessi):
            self.blockKernel.append(define_matern_kernel(coef,nu,el))
        
        self.commonKernel=define_matern_kernel(self.coeffAll,
                                               self.smoothnessAll,
                                               self.lengthAll)
            
    def update_block_kernels(self,noisei=None,
                             mui=None,
                             lengthi=None,
                             coeffi=None,
                             smoothnessi=None):
        """Update each block kernel with new parameters."""
        if noisei is None:
            noisei=self.noisei
        else:
            self.noisei=noisei
        if mui is None:
            mui=self.mui
        else:
            self.mui=mui
        if lengthi is None:
            lengthi=self.lengthi
        else:
            self.lengthi=lengthi
        if coeffi is None:
            coeffi=self.coeffi
        else:
            self.coeffi=coeffi
        if smoothnessi is None:
            smoothnessi=self.smoothnessi
        else:
            self.smoothnessi=smoothnessi
        
        for i in xrange(self.n):
            self.blockKernel[i]=define_matern_kernel(coeffi[i],
                                                     smoothnessi[i],
                                                     lengthi[i])
        self._define_kernel()
            
    def update_common_kernel(self,noise=None,
                             mu=None,
                             length=None,
                             coeff=None,
                             smoothness=None):
        """Update common kernel with new parameters."""
        if noise is None:
            noise=self.noiseAll
        else:
            self.noiseAll=noise
        if mu is None:
            mu=self.muAll
        else:
            self.muAll=mu
        if length is None:
            length=self.lengthAll
        else:
            self.lengthAll=length
        if coeff is None:
            coeff=self.coeffAll
        else:
            self.coeffAll=coeff
        if smoothness is None:
            smoothness=self.smoothnessAll
        else:
            self.smoothnessAll=smoothness
            
        self.commonKernel=define_matern_kernel(coeff,
                                               smoothness,
                                               length)
        self._define_kernel()
    
    def _define_kernel(self):
        """Define kernel function with current set of block and common kernels into
        self.kernel. Update under-the-hood GP.
        """
        def k(x,y,
              noise=self.noiseAll,
              dist=self.dist,
              commonKernel=self.commonKernel,
              blockKernel=self.blockKernel):
            # Calculate the underlying kernel for all subjects.
            d=dist(x[1:],y[1:])
            cov=commonKernel(d)

            # Add noise if the windows are the same.
            if np.isclose(d,0,rtol=0,atol=1e-15):
                cov+=noise**2

            # Add individual specific covariance.
            if x[0]==y[0]:
                cov+=blockKernel[int(x[0])](d)
                if np.isclose(d,0,rtol=0,atol=1e-15):
                    cov+=noise**2
            return cov
        
        self.kernel=k
        
        # Update GPR under-the-hood that relies on this kernel.
        self._update_gp()
        
    def train(self,X,Y):
        assert X.ndim==2
        Y=Y.copy()
        
        # Subtract predicted means.
        for i,x in enumerate(X):
            Y[i]-=self.mui[int(x[0])]
        
        self.gp.fit(X,Y)

    def predict(self,X):
        assert X.ndim==2
        Y=self.gp.predict(X)
        
        # Account for predicted means.
        for i,x in enumerate(X):
            Y[i]+=self.mui[int(x[0])]
            
        return Y
    
    def optimize_hyperparameters(self,X,Y,
                                 initial_guess=None,
                                 common=True,
                                 block=True,
                                 return_full_output=False):
        """Optimize hyperparameters. Can optimize block specific parameters and common 
        parameters together or separately.
        
        Parameters
        ----------
        X : ndarray
        Y : ndarray
        initial_guess : ndarray
        common : bool,True
            If True, optimize common hyperparameters shared between blocks.
        block : bool,True
            If True, optimize block specific hyperparameters.
        return_full_output : bool,True
            Switch for returning output of scipy.optimize.minimize.
            
        Returns
        -------
        minimize_output : dict
        """
        if initial_guess is None:
            initial_guess=np.ones(5+self.n*5)
        else:
            assert len(initial_guess)==(5+self.n*5)
        
        # Save current state.

        # Set up optimization.
        # The first 5 parameters are for the common kernel.
        if common and block:
            def update_parameters(params):
                params=list(params)
                assert len(params)==(5+5*self.n)
                
                paramsCo=params[:5]
                del params[:5]
                if not self._check_common_params(paramsCo): return False
                
                paramsBlock=[]
                for i in xrange(5):
                    paramsBlock.append(np.array(params[:self.n]))
                    del params[:self.n]
                if not self._check_block_params(paramsBlock): return False
                
                self.update_common_kernel(*paramsCo)
                self.update_block_kernels(*paramsBlock)
                return True
                
        elif common:
            def update_parameters(paramsCo):
                assert len(paramsCo)==5
                if not self._check_common_params(paramsCo): return False
                self.update_common_kernel(*paramsCo)
                return True
                
        elif block:
            def update_parameters(params):
                assert len(params)==(5*self.n)
                paramsBlock=[]
                for i in xrange(5):
                    paramsBlock.append(params[:self.n])
                    del params[:self.n]
                if not self._check_block_params(paramsBlock): return False
                
                self.update_block_kernels(*paramsBlock)
                return True
                
        def neg_log_L(params):
            if not update_parameters(params): return 1e30
            self.train(X,Y)
            return -self.gp.log_likelihood()
                
        # Run optimization.
        soln=minimize(neg_log_L,initial_guess)
        update_parameters(soln['x'])
        if return_full_output:
            return soln
        
    def print_params(self):
        print "Common parameters"
        print "Noise: %1.2f"%self.noiseAll
        print "Mean: %1.2f"%self.muAll
        print "Length: %1.2f"%self.lengthAll
        print "Coeff: %1.2f"%self.coeffAll
        print "Smoothness: %1.2f"%self.smoothnessAll
        print ""
        
        print "Block parameters"
        print "Noise:"
        print self.noisei
        print "Mean:"
        print self.mui
        print "Length:"
        print self.lengthi
        print "Coeff:"
        print self.coeffi
        print "Smoothness:"
        print self.smoothnessi

    @staticmethod
    def _check_common_params(paramsCo):
        if ( paramsCo[0]<=0 or 
             paramsCo[2]<=0 or
             paramsCo[3]<=0 or 
             paramsCo[4]<=0 ): return False
        return True
    
    @staticmethod
    def _check_block_params(paramsBlock):
        if ( (paramsBlock[0]<=0).any() or
             (paramsBlock[2]<=0).any() or
             (paramsBlock[3]<=0).any() or
             (paramsBlock[4]<=0).any() ): return False
        return True
#end BlockGPR



class GaussianProcessRegressor(object):
    def __init__(self,kernel,beta,approximate_cov=None):
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
        self.approximate_cov=approximate_cov

        self.cov = None
        self.invCov = None
        self.X = None
        self.Y = None

    def fit(self,X,Y):
        """
        This only involves calculating the covariance matrix. The output Y is only taken in so that
        it can be saved into this instance.

        Parameters
        ----------
        X : ndarray
            n_samples x n_dim. Input.
        Y : ndarray
            Measured target variable.
        """
        assert X.ndim==2
        assert len(X)==len(Y)

        self.X,self.Y = np.array(X),np.array(Y)
        self.cov = self.calc_cov(X)
        self.invCov = self.invert_cov(self.cov,self.approximate_cov)
    
    @staticmethod
    def invert_cov(cov,approximate_cov=None):
        """Invert covariance matrix. Implements approximation schemes.

        Parameters
        ----------
        cov : ndarray
        approximate_cov : tuple,None

        Returns
        -------
        invCov : ndarray
        """
        if approximate_cov is None:
            icov = np.linalg.inv(cov)

        elif approximate_cov[0]=='low rank' or approximate_cov[0]=='lr':
            rank=approximate_cov[1]

            # Eigendecomposition.
            el,v=np.linalg.eig(cov)
            keepIx=np.argsort(np.abs(el))[::-1][:rank]
            
            # Construct low-rank approximation.
            diag=np.eye(rank)*el[keepIx]
            approxCov=v[:,keepIx].dot(diag).dot(v[:,keepIx].T)

            print np.linalg.norm(approxCov-cov)
            
            icov=np.linalg.inv(approxCov)

        else: raise Exception("Unavailable approximation scheme for covariance inversion.")

        return icov

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
        verbose : bool,True

        Returns
        -------
        mu : ndarray
        err : ndarray
            Standard deviation of Gaussian process.
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
                cov[i,i] += 1/self.beta + kernel(X[i],X[i])
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

        # Covariance matrix should be symmetric.
        if np.linalg.norm(self.cov-self.cov.T)!=0:
            print "ERR: Covariance matrix is not symmetrix."
        else:
            print "OK: Covariance matrix is symmetric."

        # Check that covariance matrix is well conditioned.
        det=np.linalg.slogdet(self.cov)
        if det[0]==-1:
            print "ERR: Determinant is negative."
        elif det[1]<-10:
            print "ALERT: Covariance matrix is ill-conditioned."
        else:
            print "OK: Covariance matrix is okay."

        # Check that errors are positive.
        mu,std=self.predict(self.X,return_std=True,verbose=False)
        if np.isnan(std).any():
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
