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


% Part 1

one_mat = ones(m, 1)'; % size(one_mat) : 1 * 5000

% [one_mat; X'] : 1 * 5000; 400 * 5000 => 401 * 5000
% size(Theta1): 25 * 401
% => z1: 25 * 5000, size(a1): 25* 5000
z1 = Theta1 * [one_mat; X'];
a1 = sigmoid(z1);

% size(z2) : [1*5000;25*5000] => 26 * 5000
% size(Theta2): 10 * 26
% size(H): 10 * 5000
z2 = Theta2 * [one_mat; a1];
H = sigmoid(z2);

% make class Y
% size(Y): m*num_label
Y = zeros(m, num_labels);
for i=1:m
    Y(i, y(i)) = 1;
end

% test code
% size(Y): 5000*10
% size(H): 10*5000
% 


% submit 1
% J = - 1/m *(sum(sum( Y .* log(H)' + (ones(size(Y))-Y) .* log(ones(size(H)) - H )')));

% submit 2
T1 = Theta1(:, 2:end);
T2 = Theta2(:, 2:end);
J = - 1/m * (sum(sum( Y .* log(H)' + (ones(size(Y))-Y) .* log(ones(size(H)) - H)'))) ...
                            + lambda / (2*m) * (sum(sum(T1 .* T1)) + sum(sum(T2 .* T2)));

% Back Propagation
for i=1:m
    % a1: 401 * 1
    % a2: 26 * 1
    % a3: 10 * 1
    % Y: 5000 * 10
    a_1 = [1; X(i,:)'];
    a_2 = [1; sigmoid(Theta1 * a_1)];
    a_3 = sigmoid(Theta2 * a_2);

    delta_3 = a_3 - Y(i,:)';
    delta_2 = (Theta2' * delta_3 .* a_2 .* (1-a_2))(2:end);
    Theta2_grad += delta_3 * a_2';
    Theta1_grad += delta_2 * a_1';
end

Theta1_grad(:, 2:end) += lambda * Theta1(:, 2:end);
Theta1_grad /= m;

Theta2_grad(:, 2:end) += lambda * Theta2(:, 2:end);
Theta2_grad /= m;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
