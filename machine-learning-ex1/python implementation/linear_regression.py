import numpy as np


def cost_func(X, y, theta):
	pred = X.dot(theta)
	J = np.sum((pred - y)**2)/(float(2*m))
	
theta = n.zeros((2,1))

def gradient_descent(X, y, theta, alpha, num_iters):
	
	J_history = np.zeros((num_iters, 1));
	
	for iter in range(num_iters):
	
		h1 = X.dot(theta)
	
		delta1 = np.sum( np.multiply( (h1-y), X[:, 0]) )/float(m)
		delta2 = np.sum( np.multiply( (h1-y), X[:, 1]) )/float(m)
	
		thetao = theta[0,1] - (delta1 * alpha)
		theta1 = theta[1,0] - (delta2 * alpha)
	
		theta = np.matrix([thetao, theta1]).reshape(-1,1)
		
		J_history.append(cost_func(X, y, theta))
	
	return J_history, theta
	

def feature_normalize(X):
	mu = np.mean(X)
	sigma = np.std(X)
	X_norm = (X-mu)/float(sigma)
	
	return X_norm, mu, sigma
	

	



