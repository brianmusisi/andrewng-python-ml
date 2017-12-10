import numpy as np  
import pandas as pd

Nfeval =0
#definition of sigmoid function which is the activation function for the neural network
def sigmoid(z):  
    return 1 / (1 + np.exp(-z))
    

#forward propagation that moves forward through the network
def forwardprop(X, theta1, theta2):  
    m = X.shape[0]

    input1 = np.insert(X, 0, values=np.ones(m), axis=1) #first input, bias term added as column of ones
    z2 = input1 * theta1.T    
    input2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1) #get the input to the second layer
    z3 = input2 * theta2.T  
    output = sigmoid(z3)  #get the final output prediction as probabilites
    return input1, z2, input2, z3, output
    
    

#create the cost function for the network that we shall aim to minimize    
def cost_function(params,input_layer_size, hidden_layer_size, num_labels,X, y):
    X = np.matrix(X) # turn X and y into matrices
    y = np.matrix(y)
    m = X.shape[0] #m is the number of examples in the dataset

	#from the params array, obtain the  initial randomized matrices for theta1 and theta2
    theta1 = np.matrix(np.reshape(params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))
    
    _, _, _, _, output = forwardprop(X, theta1, theta2) # perform forward propagation and obtain the output matrix

    J = 0     #initiliaze the cost, J to 0
    theta1_grad = np.zeros(theta1.shape)  #initialize the gradients of the theta matrices to all zeros
    theta2_grad = np.zeros(theta2.shape)   
    
    y_k = y.reshape(m * num_labels, -1)  #reshape y, flatten it to a 1D array
    h = output.reshape(m*num_labels,-1)  #rehsape the output to a 1D array
    J_k = np.reshape((np.multiply(-y_k, np.log(h)) - (np.multiply( (1-y_k) , np.log(1-h) ) ) ), (m, -1)) #vectorized calculation of the cost
    reg_cost = (float(lam)/(2*m))*(np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2))) #regluarized portion of the cost
    J = np.sum(J_k)/m + reg_cost # sum up the cost matrix and add the regularized cost
    return J #return the cost

  
    
def sigmoid_gradient(z):  
    return np.multiply(sigmoid(z), (1 - sigmoid(z))) #formula to obtain the gradient of the sigmoid function


    
def backpropagation(params,input_layer_size, hidden_layer_size, num_labels,X, y, lam):
	#first portion is identical to the cost function above
    X = np.matrix(X)
    y = np.matrix(y)
    m = X.shape[0]

    theta1 = np.matrix(np.reshape(params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))

    input1, z2, input2, z3, output = forwardprop(X, theta1, theta2)

    J = 0
    theta1_grad = np.zeros(theta1.shape)
    theta2_grad = np.zeros(theta2.shape)   

    y_k = y.reshape(m * num_labels, -1)
    h = output.reshape(m*num_labels,-1)
    J_k = np.reshape((np.multiply(-y_k, np.log(h)) - (np.multiply( (1-y_k) , np.log(1-h) ) ) ), (m, -1))
    reg_cost = (float(lam)/(2*m))*(np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    J = np.sum(J_k)/m + reg_cost
	
	#implementation of the backpropagation algorithm
    
    y_n = y.T # for easier use of the y matrix for vectorized computations(avoiding loops), we transpose it

    for row in range(m):
    	#the first of this for-loop is forward propagation broken down for a particular row. Look at forwardprop function above
    	#the tuples at the end show the matrix shapes
        a1_row = input1[row,:]  # (1, 401)
        z2_row = theta1 * a1_row.T  # (25,1)   )
        a2_row = sigmoid(z2_row)  # (25,1)
        a2_row = np.concatenate((np.ones((1,a2_row.shape[1])), a2_row),axis=0)  
        z3_row= theta2 * a2_row   #(10,1)    
        h_row = sigmoid(z3_row)   #(10,1)
        y_row = y_n[:, row].reshape(y_n.shape[0],-1)     #(10, 1)
        
		#below is the backpropagation algorithm broken down
        d3_row = h_row - y_row  # (10, 1) #first get the difference between prediction and actual values

        z2_row = np.insert(z2_row, 0, values=np.ones(1),axis=0)  #(26,1)  Add bias row to the matrix
        d2_row = np.multiply((theta2.T * d3_row ), sigmoid_gradient(z2_row))  # (26, 1)  obtain difference between values in the first layer

        d2_row = d2_row[1:] #(25, 1)   #remove the bias term

        theta1_grad = theta1_grad + d2_row * a1_row  #(25,401)  calculate the gradient for between the first two layers
        theta2_grad = theta2_grad + d3_row * a2_row.T #(10,26)  calculate the gradient between the second set of layers

    reg_theta1 = theta1 * (lam/float(m)) #obtain regularized gradient and remove first bias term(make it zero)
    reg_theta1[:, 0] = 0

    reg_theta2 = theta2 * (lam/float(m)) #obtain regularized gradient and remove first bias term(make it zero)
    reg_theta2[:, 0] = 0

    # add the gradient regularization term to the inital gradient
    theta1_grad = theta1_grad /m + reg_theta1  
    theta2_grad = theta2_grad /m + reg_theta2

    # unravel the gradient matrices and concatenate them into a single
    grad = np.concatenate((np.ravel(theta1_grad), np.ravel(theta2_grad)))
    
    return J, grad   # return the cost together with the gradient
 
 
#this functions trains and fits the model, returning the optimum values of theta1 and theta2 that are used   
def fit_model(X, y, hidden_layer_size=25, epsilon=0.2, learning_rate=1):
	from sklearn.preprocessing import OneHotEncoder  #use one hot encoding on the y array
	encoder = OneHotEncoder(sparse=False)
	y = y.reshape(y.shape[0], -1)
	y_onehot = encoder.fit_transform(y)  
	input_layer_size = X.shape[1]  #number of features in the input
	hidden_layer_size = hidden_layer_size  #size of the hidden layer
	num_labels = len(np.unique(np.array(y))) # number of labels in the output
	
	#params variable that holds the optimal parameters for the model(combines theta1 and theta2)
	#here we randomly select values for it to be assigned to the theta values, adjustable using epsilon
	params = (np.random.random(size=hidden_layer_size * (input_layer_size + 1) + num_labels * (hidden_layer_size + 1)) - (2*epsilon)) * epsilon
	
	#we use scipy's minimize function that will try to find the values of params that minimize the cost J we pass to it using the gradient grad for the parameters
	from scipy.optimize import minimize
	
	Nfeval=0
	
	def callbackF(xk):
		global Nfeval
		Nfeval+=1
		print 'Iteration %i completed'%Nfeval
	
	#fitting the parameters
	fmin = minimize(fun=backpropagation, x0=params, args=(input_layer_size, hidden_layer_size, num_labels, X, y_onehot, learning_rate),method='TNC', jac=True, options={'maxiter': 250}, callback=callbackF)
	
	#assign the optimal values of params that have been obtained to theta1 and theta2
	theta1 = np.matrix(np.reshape(fmin.x[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))
	theta2 = np.matrix(np.reshape(fmin.x[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))
	
	return theta1, theta2 #return the optimal values of theta
	
	
#the function that makes the prediction and calculates the accuracy
def predict(theta1, theta2, X, y):
	_, _, _, _, output = forwardprop(X, theta1, theta2) #perform forward propagation
	y_pred = np.array(np.argmax(output, axis=1) + 1) #for each row obtain the position of the highest probability as the prediction(add 1 because arrays are zero-indexed)
	
	accuracy = np.sum(np.array((y_pred==y)))/float(len(y)) *100 #calculate the accuracy
	
	print 'The accuracy is %.3f' %accuracy  #print the accuracy
	return y_pred
	


	
	