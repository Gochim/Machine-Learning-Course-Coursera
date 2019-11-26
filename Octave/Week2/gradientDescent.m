function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    prediction = X * theta;
    difference = prediction - y;
    
    % variant 1
%     delta = zeros( length(X(1, :)), 1 );
%     for i = 1:m
%         delta = delta + difference(i) * X(i, :)';
%     end
%     delta = 1/m * delta;

    ## variant 2
##    tdiff = difference';
##    delta = 1/m * tdiff * X;
##    theta = theta - alpha * delta';

    % variant 3
    theta = theta - alpha / m * X' * difference;

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
