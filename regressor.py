# =============================================================================================== #
# Module for Gaussian process regression.
# Author: Eddie Lee, edlee@alumni.princeton.edu
# =============================================================================================== #

from __future__ import division
import numpy as np
from numba import jit
from scipy.spatial.distance import pdist,squareform
from itertools import combinations
from scipy.optimize import minimize,minimize_scalar



class BlockGPR(object):
    """
    Gaussian process regression with blocked covariance matrix. There are covariance terms specific
    to each block and a shared common covariance term across all blocks.

    This relies on GaussianProcessRegressor for the underlying GPR model.
    """
    def __init__(self,n,dist=None):
        """
        Parameters
        ----------
        n : int
            Number of blocks.
        dist : function,None
            Distance function to use in symmetric kernel.
        """
        self.n=n  # no. of disjoint blocks
        self.dist=dist or (lambda x,y:np.linalg.norm(x-y))
        
        # Default block specific parameters.
        self.noisei=np.zeros(n)+.5
        self.mui=np.zeros(n)
        self.lengthi=np.zeros(n)+1
        self.coeffi=np.ones(n)
        self.smoothnessi=np.zeros(n)+.5

        # Default parameters on common landscape shared by all block kernels.
        self.noiseCo=.5
        self.muCo=0.
        self.lengthCo=1
        self.coeffCo=1
        self.smoothnessCo=.5

        # Noise term per data entry. This is independent noise per experimental run even if on the
        # same block with the same parameters.
        self.noise=.2
        
        self._setup_kernels()
        self._define_kernel()
        
        self._update_gp()
        
    def _update_gp(self,X=None):
        self.gp=GaussianProcessRegressor(self.kernel,1/self.noise,
             custom_noise_function=lambda x,y,i,j,noisei=self.noisei: noisei[x[0]]**2 if i==j else 0)
    
    def _setup_kernels(self):
        """First time instance is declared."""
        self.blockKernel=[]
        for el,coef,nu in zip(self.lengthi,self.coeffi,self.smoothnessi):
            self.blockKernel.append(define_matern_kernel(coef,nu,el))
        
        self.commonKernel=define_matern_kernel(self.coeffCo,
                                               self.smoothnessCo,
                                               self.lengthCo)

    def update_block_kernels(self,
                             lengthi=None,
                             coeffi=None,
                             smoothnessi=None):
        """Update each block kernel with new parameters."""
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
            
    def update_common_kernel(self,
                             length=None,
                             coeff=None,
                             smoothness=None):
        """Update common kernel with new parameters."""
        if length is None:
            length=self.lengthCo
        else:
            self.lengthCo=length
        if coeff is None:
            coeff=self.coeffCo
        else:
            self.coeffCo=coeff
        if smoothness is None:
            smoothness=self.smoothnessCo
        else:
            self.smoothnessCo=smoothness
            
        self.commonKernel=define_matern_kernel(coeff,
                                               smoothness,
                                               length)
        self._define_kernel()

    def update_noise(self,noise=None,noisei=None):
        if noise is None:
            noise=self.noise
        else:
            self.noise=noise
        if noisei is None:
            noisei=self.noisei
        else:
            self.noisei=noisei
        self._update_gp()

    def _define_kernel(self):
        """Define kernel function with current set of block and common kernels into
        self.kernel. Update under-the-hood GP.
        """
        def k(x,y,
              dist=self.dist,
              commonKernel=self.commonKernel,
              blockKernel=self.blockKernel):
            # Geodesic distance. 
            d=dist(x[1:],y[1:])

            # Calculate the underlying kernel for all subjects.
            cov=commonKernel(d)

            # Add individual specific covariance.
            if x[0]==y[0]:
                cov+=blockKernel[int(x[0])](d)
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
                                 fix_params_function=None,
                                 reg_function=None,
                                 check_bounds_function=None,
                                 minimize_kw={},
                                 return_full_output=False,
                                 verbose=False):
        """Optimize hyperparameters. Can optimize block specific parameters and common 
        parameters together or separately.

        Sets self.kernel and resets kernel for self.gp. self.gp has been reset. This means that you
        should retrain it.
        
        Parameters
        ----------
        X : ndarray
        Y : ndarray
        initial_guess : ndarray
        common : bool,True
            If True, optimize common hyperparameters shared between blocks.
        block : bool,True
            If True, optimize block specific hyperparameters.
        fix_params_function : function,None
            If a function is passed, this can be used to fix particular parameters to certain
            values. This is an easy way to customize the parameter optimization code without making
            it overly complicated here.

            This will be called in update_parameters() at each iteration of the minimization
            routine. Parameter list should be changed in place.

            If both common and block switches are set, the first 5 parameters are for the common
            kernel, each sequential set of self.n correspond to the block kernels and the last
            parameter is the common noise term. All the sets of parameters are ordered as noise, mu,
            length, coeff, smoothness.
        reg_function : function,None
            Function for imposing regularization on the optimization given parameters.
        check_bounds_function : function,None
            Function that returns True if all parameters are within bounds. This is run immediately
            after new set of parameters are proposed, so it can be much faster to run checks on
            parameters in this function rather than in reg_function.
        return_full_output : bool,True
            Switch for returning output of scipy.optimize.minimize.
            
        Returns
        -------
        minimize_output : dict
        """
        # Check inputs.
        assert X.ndim==2 and Y.ndim==1
        assert common or block
        if initial_guess is None:
            if common and block:
                initial_guess=np.ones(5+self.n*5+1)/3
            elif common:
                initial_guess=np.ones(5+1)/3
            elif block:
                initial_guess=np.ones(self.n*5+1)/3
        else:
            if common and block:
                assert len(initial_guess)==(5+self.n*5+1)
            elif common:
                assert len(initial_guess)==(5+1)
            elif block:
                assert len(initial_guess)==(self.n*5+1)
        if fix_params_function is None:
            fix_params_function=lambda x:None
        fix_params_function(initial_guess)
        if reg_function is None:
            reg_function=lambda x:0.
        if check_bounds_function is None:
            check_bounds_function=lambda x:True

        # Save current state.

        # Set up optimization.
        # The first 5 parameters are for the common kernel. The following parameters are for the
        # block kernels in sets of 5. The final parameter is the common noise term on the diagonal
        # of the entire covariance matrix.
        if common and block:
            def update_parameters(params):
                fix_params_function(params)
                params=list(params)
                
                paramsCo=params[:5]
                del params[:5]
                if not self._check_common_params(paramsCo): return False
                
                paramsBlock=[]
                for i in xrange(5):
                    paramsBlock.append(np.array(params[:self.n]))
                    del params[:self.n]
                if not self._check_block_params(paramsBlock): return False
                
                if params[-1]<=0: return False
                self.update_noise(params[-1],paramsBlock[0])

                self.update_common_kernel(*paramsCo[2:])
                self.update_block_kernels(*paramsBlock[2:])
                return True
                
        elif common:
            def update_parameters(paramsCo):
                fix_params_function(paramsCo)
                params=list(params)

                if not self._check_common_params(paramsCo): return False

                if params[-1]<=0: return False
                self.update_noise(params[-1])

                self.update_common_kernel(*paramsCo[2:])
                return True
                
        elif block:
            def update_parameters(params):
                fix_params_function(params)
                params=list(params)

                paramsBlock=[]
                for i in xrange(5):
                    paramsBlock.append(params[:self.n])
                    del params[:self.n]
                if not self._check_block_params(paramsBlock): return False
                
                if params[-1]<=0: return False
                self.update_noise(params[-1],paramsBlock[0])

                self.update_block_kernels(*paramsBlock[2:])
                return True
        
        if verbose:
            def neg_log_L(params):
                if not check_bounds_function(params):return 1e30
                if not update_parameters(params): return 1e30
                self.train(X,Y)
                L=self.gp.neg_log_likelihood()+reg_function(params)
                print "Cost=%1.3f"%L
                return L
        else:
            def neg_log_L(params):
                if not check_bounds_function(params):return 1e30
                if not update_parameters(params): return 1e30
                self.train(X,Y)
                return self.gp.neg_log_likelihood()+reg_function(params)

        # Run optimization.
        soln=minimize(neg_log_L,initial_guess,**minimize_kw)
        update_parameters(soln['x'])

        if return_full_output:
            return soln
        
    def print_params(self):
        print "Common parameters"
        print "Noise: %1.2f"%self.noiseCo
        print "Mean: %1.2f"%self.muCo
        print "Length: %1.2f"%self.lengthCo
        print "Coeff: %1.2f"%self.coeffCo
        print "Smoothness: %1.2f"%self.smoothnessCo
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

        print ""
        print "Diag: %1.2f"%self.noise

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
    def __init__(self,kernel,beta,
                 approximate_cov=None,
                 custom_noise_function=lambda x,y,xi,yi:0):
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
        approximate_cov : tuple,None
            This can trigger approximation schemes for the covariance matrix. See invert_cov()
            method.
        custom_noise_function : function,lambda x,y,xi,yi:0
            Function for adding custom entries into the covariance matrix. For example, this could
            be used to add diagonals to block matrices when we have a block matrix. Args are the two
            data points x and y and their indices in the covariance matrix xi and yi. Whatever is
            returned by this function is added to the corresponding entry in the covariance matrix.
        """
        assert beta>0

        self.kernel = kernel
        self.beta = beta
        self.noise_function=custom_noise_function
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
        def calc_cov(X,kernel=self.kernel,noise_function=self.noise_function):
            nSamples = len(X)
            cov = np.zeros((nSamples,nSamples))
            for i,j in combinations(range(nSamples),2):
                cov[i,j] = cov[j,i] = kernel(X[i],X[j]) + noise_function(X[i],X[j],i,j)
            for i in xrange(nSamples):
                cov[i,i] += 1/self.beta + kernel(X[i],X[i])
            return cov

        self.calc_cov = calc_cov
   
    def neg_log_likelihood(self):
        """Negative log-likelihood of the current state of the GPR.
        """
        return -self.log_likelihood() 

    def log_likelihood(self):
        """Log-likelihood of the current state of the GPR.
        """
        det=np.linalg.slogdet(self.cov)
        assert det[0]>0,(det,self.cov.min())
        return ( -.5*det[1]
                 -.5*self.Y.dot(np.linalg.inv(self.cov)).dot(self.Y)
                 -len(self.X)/2*np.log(2*np.pi) )

    def sample(self,cov=None,size=()):
        from numpy.random import multivariate_normal
        if cov is None:
            cov=self.cov
        return multivariate_normal( np.zeros(len(cov)),cov,size=size )

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
            print "OK: Co errors are positive."

    def _noise_matrix(self,X):
        """
        Calculate the entries of the noise matrix. This is primarily for testing since it's more
        efficient to only loop through the entries of the covariance matrix once.
        """
        noiseMat=np.zeros((len(X),len(X)))
        for i in xrange(len(X)-1):
            for j in xrange(i,len(X)):
                noiseMat[j,i]=noiseMat[i,j]=self.noise_function(X[i],X[j],i,j)
        return noiseMat
#end GaussianProcessRegressor



# =============================================================================================== #
# Useful functions.
# =============================================================================================== #
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

def define_matern_kernel(coeff,nu,c):
    """
    Define a function for calculating the Matern kernel.

    Parameters
    ----------
    nu : float
        Smoothness parameter. When nu->inf, we have the RBF kernel.
    c : float
        Length scale.
    
    Returns
    -------
    matern_kernel : function
        Takes list of distances x. Keyword args nu and c set to given default values.
    """
    from scipy.special import gamma,kv
    assert c>0
    assert nu>0
    def f(x,nu=nu,c=c):
        x=np.array(x)
        if x.shape==():
            x=np.array([x])
        y=coeff*2**(1-nu)/gamma(nu)*(x/c)**nu * kv(nu,x/c)
        y[x==0]=coeff
        
        return y
    return f
