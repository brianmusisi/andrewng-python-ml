import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

def cost_function(params, Y, R, num_users, num_movies, num_features, lam):

	X = np.reshape(params[: num_movies* num_features] , (num_movies, num_features))
	Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features))
	J=0
	
	X_grad = np.zeros_like(X)
	Theta_grad = np.zeros_like(Theta)
	
	z= np.dot(X, Theta.T)
	
	J =   np.sum(np.multiply(((z - Y)**2), R))/2 +  (np.sum((Theta **2)) + np.sum((X**2)))*(lam/2)
	
	X_grad =  np.dot(np.multiply((z-Y), R), Theta) + (lam*X)
	
	Theta_grad = np.dot((np.multiply((z-Y), R)).T, X) + (lam * Theta)
	
	grad = np.concatenate((np.ravel(X_grad), np.ravel(Theta_grad)))
	
	
	#J = sum(sum( ((z-Y) .^2) .*R) ) /2  + (sum(sum((Theta.^2),2)) + sum(sum((X.^2),2)))*(lambda/2);
	
	return J, grad
	
	
def initialize_weights(num_users, num_movies, num_features):
	params = np.random.random(num_movies* num_features + num_users * num_features)
	return params
	
	
def collab_fit(num_features = 12, lam = 1.5):
	data = loadmat('ex8_movies.mat')
	
	Y = data['Y']
	R = data['R']
	
	num_movies, num_users = Y.shape[0], Y.shape[1]
	
	
	params = np.random.random(num_movies* num_features + num_users * num_features)
	
	fmin = minimize(fun=cost_function, x0=params, args=(Y, R, num_users, num_movies, num_features, lam),method='TNC', jac=True, options={'maxiter': 250})
	
	X = np.reshape(fmin.x[: num_movies* num_features] , (num_movies, num_features))
	Theta = np.reshape(fmin.x[num_movies*num_features:], (num_users, num_features))
	
	return X, Theta
	
	
if __name__ == '__main__':
	collab_fit()
	
	
	
	
	
