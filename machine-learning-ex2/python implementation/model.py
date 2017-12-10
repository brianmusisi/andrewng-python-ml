import reg


#First run optimize below with the train data, features as X and target as y
#then use the returned values of theta to make predictions on the test data
def optimize(X, y):
	m,n = X.shape

	best_cost =200
	best_lam=20
	best_theta = np.zeros(n)
	lams =[0.01, 0.03, 0.05, 0.1, 0.3, 0.5, 1, 10]

	for lam in lams:
	    initial_theta = np.zeros(n)
	    Result = reg.op.minimize(fun = reg.CostFunc, x0 = initial_theta, args = (X, y, lam),method = 'TNC',jac = reg.Gradient);
	    optimal_theta = Result.x
	    cost = reg.CostFunc(optimal_theta, Xval, yval, lam)
	    if cost<best_cost:
	        best_cost = cost
	        best_lam = lam
	        best_theta = optimal_theta
	 return best_theta

#use best_theta from the ptimize function to make predictions on the test data
def predict(X_test, best_theta):
	y_test = reg.Sigmoid(X_test.dot(best_theta))
	y_test[y_test<0.5]=0
	y_test[y_test>0.5]=1

	return y_test