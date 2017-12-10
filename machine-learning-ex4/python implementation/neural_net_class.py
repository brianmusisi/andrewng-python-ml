import numpy as np  
import pandas as pd


class NeuralNet(object):
    def __init__(self, X):
    	input_layer_size = X.shape[1]  
        hidden_layer_size = hidden_layer_size  
        num_labels = len(np.unique(np.array(y)))
        
        params = (np.random.random(size=hidden_layer_size * (input_layer_size + 1) + num_labels * (hidden_layer_size + 1)) - (2*epsilon)) * epsilon
        
        theta1 = np.matrix(np.reshape(params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))
        theta2 = np.matrix(np.reshape(params[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))
            
    def sigmoid(self, z):  
        return 1 / (1 + np.exp(-z))

    def forwardprop(self,X, theta1, theta2):  
        m = X.shape[0]

        input1 = np.insert(X, 0, values=np.ones(m), axis=1)
        z2 = input1 * theta1.T
        input2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
        z3 = input2 * theta2.T
        output = self.sigmoid(z3)

        return input1, z2, input2, z3, output
        
    def cost_function(self, params,input_layer_size, hidden_layer_size, num_labels,X, y, lam):
        X = np.matrix(X)
        y = np.matrix(y)
        m = X.shape[0]

        self.theta1 = np.matrix(np.reshape(params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))
        self.theta2 = np.matrix(np.reshape(params[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))
        
        input1, z2, input2, z3, output = self.forwardprop(X, self.theta1, self.theta2)

        J = 0
        theta1_grad = np.zeros(self.theta1.shape)
        theta2_grad = np.zeros(self.theta2.shape)   
        
        y_k = y.reshape(m * num_labels, -1)
        h = output.reshape(m*num_labels,-1)
        J_k = np.reshape((np.multiply(-y_k, np.log(h)) - (np.multiply( (1-y_k) , np.log(1-h) ) ) ), (m, -1))
        reg_cost = (float(lam)/(2*m))*(np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
        J = np.sum(J_k)/m + reg_cost
        
        
    def sigmoid_gradient(self, z):  
        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))
        
    def backpropagation(self,params,input_layer_size, hidden_layer_size, num_labels,X, y, lam):
        X = np.matrix(X)
        y = np.matrix(y)
        m = X.shape[0]

        theta1 = np.matrix(np.reshape(params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))
        theta2 = np.matrix(np.reshape(params[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))
        

        input1, z2, input2, z3, output = self.forwardprop(X, self.theta1, self.theta2)

        J = 0
        theta1_grad = np.zeros(self.theta1.shape)
        theta2_grad = np.zeros(self.theta2.shape)   

        y_k = y.reshape(m * num_labels, 1)
        h = output.reshape(m*num_labels,1)
        J_k = np.reshape((np.multiply(-y_k, np.log(h)) - (np.multiply( (1-y_k) , np.log(1-h) ) ) ), (m, -1))
        reg_cost = (float(lam)/(2*m))*(np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
        J = np.sum(J_k)/m + reg_cost

        y_n = y.T

        for t in range(m):
            a1t = input1[t,:]  # (1, 401)
            z2t = theta1 * a1t.T  # (25,1)   theta1 : (25:401)
            a2t = self.sigmoid(z2t)  # (25,1)
            a2t = np.concatenate((np.ones((1,a2t.shape[1])), a2t),axis=0)  #(26, 1)
            z3t= theta2 * a2t   #(10,1)    theta2 : (10, 26)
            ht = self.sigmoid(z3t)   #(10,1)
            yt = y_n[:, t].reshape(y_n.shape[0],-1)     #(10, 1)
            

            d3t = ht - yt  # (10, 1)

            z2t = np.insert(z2t, 0, values=np.ones(1),axis=0)  #(26,1)
            d2t = np.multiply((theta2.T * d3t ), self.sigmoid_gradient(z2t))  # (26, 1)

            d2t = d2t[1:] #(25, 1)

            theta1_grad = theta1_grad + d2t * a1t  #(25,401)
            theta2_grad = theta2_grad + d3t * a2t.T #(10,26)

        reg_theta1 = theta1 * (lam/float(m))
        reg_theta1[:, 0] = 0

        reg_theta2 = theta2 * (lam/float(m))
        reg_theta2[:, 0] = 0

        # add the gradient regularization term
        theta1_grad = theta1_grad /m + reg_theta1
        theta2_grad = theta2_grad /m + reg_theta2

        # unravel the gradient matrices into a single array
        grad = np.concatenate((np.ravel(theta1_grad), np.ravel(theta2_grad)))
        
        print 'Cost is %.3f' %J

        return J, grad
        
    def fit_model(self, X, y, hidden_layer_size=25, epsilon=0.2, learning_rate=1):
        from sklearn.preprocessing import OneHotEncoder  
        encoder = OneHotEncoder(sparse=False)  
        y_onehot = encoder.fit_transform(y)  
        input_layer_size = X.shape[1]  
        hidden_layer_size = hidden_layer_size  
        num_labels = len(np.unique(np.array(y)))
        
        params = (np.random.random(size=hidden_layer_size * (input_layer_size + 1) + num_labels * (hidden_layer_size + 1)) - (2*epsilon)) * epsilon
        
        
        from scipy.optimize import minimize
        
        fmin = minimize(fun=self.backpropagation, x0=params, args=(input_layer_size, hidden_layer_size, num_labels, X, y_onehot, learning_rate),method='TNC', jac=True, options={'maxiter': 250})
        self.theta1 = np.matrix(np.reshape(fmin.x[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))
        self.theta2 = np.matrix(np.reshape(fmin.x[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))
        
        return self.theta1, self.theta2
        
        

    def predict(self, X, y):
        theta1 = self.theta1
        theta2 = self.theta2
        
        input1, z2, input22, z3, output = forwardprop(X, theta1, theta2)
        y_pred = np.array(np.argmax(output, axis=1) + 1)
        
        accuracy = np.sum(np.array((y_pred==y)))/float(len(y)) *100
        
        print 'The accuracy is %.3f' %accuracy
        
        return y_pred