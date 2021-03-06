function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m,1), X];

z2 = Theta1 * transpose(X);

a2 = sigmoid(z2);

a2=cat(1, transpose(ones(m, 1)), a2);

h = sigmoid(Theta2 * a2);

h= reshape(h, m*num_labels,1);

y_k = zeros(m, num_labels);   %y_n is the y that is useful for the neural network where each result is a vector of the results


for iter = 1:m

y_k(iter, y(iter)) = 1;
y_n = y_k;

end


y_k = reshape(transpose(y_k), m*num_labels, 1);
J_k = reshape((-y_k.*log(h) -((1-y_k).*log(1-h)) ), m, num_labels);
reg_cost = (lambda/(2*m))*(sum(sum((Theta1(:, 2:end).^2),2)) + sum(sum((Theta2(:, 2:end).^2),2)));
J = sum(sum(J_k))/m + reg_cost;

%y_k = transpose(reshape(y_k, m, num_labels));

y_n = transpose(y_n);

%Delta1 = zeros(size(Theta1));
%Delta2 = zeros(size(Theta2));

for iter = 1:m

a1 = X(iter, :);

z2 = Theta1 * transpose(a1);

a2 = sigmoid(z2);

%a2 = cat(1, ones((size(a2,2), 1);
a2 = [1; a2];
z3 = Theta2 * a2;
a3 = sigmoid(z3);

d3 = a3 - y_n(:,iter);

d2 = (transpose(Theta2) * d3).*sigmoidGradient([1; z2]);

d2 = d2(2:end);

Theta2_grad =  Theta2_grad + d3 * transpose(a2);

Theta1_grad =  Theta1_grad + d2 * a1;

end

reg_theta1 = Theta1.*(lambda/m);
reg_theta1(:,1) = 0;

reg_theta2 = Theta2.* (lambda/m);
reg_theta2(:,1) =0;


Theta1_grad = Theta1_grad./m + reg_theta1;

Theta2_grad = Theta2_grad./m + reg_theta2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
