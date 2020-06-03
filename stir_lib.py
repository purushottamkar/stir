from sklearn.base import BaseEstimator
import numpy as np
import statsmodels.api as sm

def HT(a,k):
    t=np.zeros(a.shape)
    if k==0:
        return t
    else:
        ind=np.argpartition(abs(a),-k, axis=None)[-k:]    
        t[ind,:]=a[ind,:]
    return t

class STIR(BaseEstimator):
	def __init__( self, eta = 1.01, alpha = 0.0, M_init = 10.0, w_init = None ):
		self.eta = eta
		self.alpha = alpha
		self.M_init = M_init
		self.w_init = w_init

	def fit( self, X, y, max_iter = 40, max_iter_w = 1 ):
		n, d = X.shape
		M = self.M_init
		self.w = self.w_init
		itr=0
		while itr < max_iter:        
			iter_w = 0
			while iter_w < max_iter_w:
				s = abs( np.dot( X, self.w ) - y )
				np.clip( s, 1 / M, None, out = s )        
				s = 1/s
				
				mod_wls = sm.WLS( y, X, weights = s )
				res_wls = mod_wls.fit()
				self.w = res_wls.params.reshape( d, 1 )
				iter_w += 1     

				if iter_w >=max_iter_w:
					break
			itr += iter_w
			M *= self.eta
		return self
	
	def predict( self, X ):
		return np.dot( X, self.w )
	
	# Return negative RMSE as score since sklearn assumes that score is something to be maximized
	def score( self, X, y ):
		n_test = X.shape[0]
		n_test_corr = int( self.alpha * n_test )
		res = y - self.predict(X)
		res_corr = HT( res, n_test_corr )
		return -np.linalg.norm( res - res_corr, 2) / np.sqrt( n_test - n_test_corr )


