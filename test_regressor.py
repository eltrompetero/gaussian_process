from __future__ import division
from regressor import *

def test_regressor():
    kernel=lambda x,y:exp(-(x-y)**2/2)
    beta=10
    gpr=GaussianProcessRegressor(kernel,beta)

    X=np.array([.9,1,1.1])[:,None]
    Y=np.ones_like(X)
    gpr.fit(X,Y)

    assert np.array_equal(gpr.cov.diagonal()[0],kernel(0,0))+1/beta
