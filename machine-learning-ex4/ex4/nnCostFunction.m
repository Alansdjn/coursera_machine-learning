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

%hx = sigmoid((sigmoid(X * Theta1')) * Theta2');
%J =

% Part 1
for i = 1:m
	%a1 = [1 X(i,:)];
	%z2 = a1 * Theta1';
	%a2 = [1 sigmoid(z2)];
	%z3 = a2 * Theta2';
	%a3 = sigmoid(z3);

	a1 = [1; X(i,:)']; % 401 * 1
	z2 = Theta1 * a1; % (25 * 401) * (401 * 1) = 25 * 1
	a2 = [1; sigmoid(z2)]; % 26 * 1
	z3 = Theta2 * a2; % (10 * 26) * (26 * 1) = 10 * 1 
	a3 = sigmoid(z3); % 10 * 1


	y_i = zeros(num_labels, 1);
	y_i(y(i), 1) = 1;

	delta3 = a3 - y_i;

	%fprintf( "size(delta3) == %f\n", size(delta3));
	%fprintf( "size(Theta2) == %f\n", size(Theta2));
	%fprintf( "size(Theta2(2:end,:)\' * delta3) == %f\n", size(Theta2(:,2:end)' * delta3));
	%fprintf( "size(sigmoidGradient(z2)) == %f\n", size(sigmoidGradient(z2)));

	delta2 = (Theta2(:,2:end)' * delta3) .* sigmoidGradient(z2);

	Theta2_grad = Theta2_grad + delta3 * a2';
	Theta1_grad = Theta1_grad + delta2 * a1';
	
	J = J + sum(-1 * y_i .* log(a3) - (1 - y_i) .* log(1 - a3));
end

J = J / m;
Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad / m;

% Regularized cost J
regularized_val = 0;
if lambda != 0
	for j = 1 : hidden_layer_size
		for k = 2 : input_layer_size + 1
			regularized_val = regularized_val + Theta1(j, k)^2;
		end
	end
	for j = 1 : num_labels
		for k = 2 : hidden_layer_size + 1
			regularized_val = regularized_val + Theta2(j, k)^2;
		end
	end
	regularized_val = (regularized_val * lambda) / (2 * m);	
end
J = J + regularized_val;

% Regularized Delta
if lambda != 0
	TmpTheta1 = Theta1;
	TmpTheta1(:, 1) = 0;
	Theta1_grad = Theta1_grad + TmpTheta1 * lambda / m;

	TmpTheta2 = Theta2;
	TmpTheta2(:, 1) = 0;
	Theta2_grad = Theta2_grad + TmpTheta2 * lambda / m;
end










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
