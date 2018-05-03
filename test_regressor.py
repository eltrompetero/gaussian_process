from __future__ import division
from regressor import *


def test_BlockGPR():
    # Example with two blocks.
    gpr=BlockGPR(2)
    X=np.random.rand(4,3)
    X[:2,0]=0
    X[2:,0]=1
    Y=np.random.normal(size=4)
    gpr.update_common_kernel(coeff=1.3)
    gpr.update_block_kernels(coeffi=np.array([.6,.7]))
    
    # Some basic kernel checks.
    assert gpr.commonKernel(0)==1.3
    assert gpr.blockKernel[0](0)==.6 and gpr.blockKernel[1](0)==.7
    assert gpr.kernel(X[0],X[0])==(1.3+.6+.5)
    assert gpr.kernel(X[2],X[2])==(1.3+.7+.5)
    
    # Check that updating kernel parameters also updates underlying GP.
    assert gpr.gp.kernel(X[0],X[0])==(1.3+.6+.5)
    assert gpr.gp.kernel(X[2],X[2])==(1.3+.7+.5)
    
    # Check that GP covariance matrix is positive definite.
    gpr.train(X,Y)
    assert (gpr.gp.cov>=0).all()
    assert (np.isnan(gpr.predict(X))==0).all()
    
    # Check that predictions are correct.
    assert np.isclose(gpr.predict(X),Y,rtol=0,atol=1e-10).all()
    
    # Generate samples from GPR and find parameters.

def test_regressor():
    kernel=lambda x,y:exp(-(x-y)**2/2)
    beta=10
    gpr=GaussianProcessRegressor(kernel,beta)

    X=np.array([.9,1,1.1])[:,None]
    Y=np.ones_like(X)
    gpr.fit(X,Y)

    assert np.array_equal(gpr.cov.diagonal()[0],kernel(0,0))+1/beta
