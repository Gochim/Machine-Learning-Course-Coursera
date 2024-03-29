function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
    
    n = size(theta);

    prediction = sigmoid(X * theta);
    difference = prediction - y;
    
    % calc the cost function
    
    J = 1/m * sum(- y .* log(prediction) - (1 - y) .* log(1 - prediction)) + lambda * sum(theta(2:n).^2) / (2*m);
   % J = 1/m * sum(- y .* log(prediction) - (1 - y) .* log(1 - prediction));
    J2 = 1/m * (- y' * log(prediction) - (1 - y)' * log(1 - prediction)) + lambda * sum(theta(2:n).^2) / (2*m);
    
    % calc the gradient
    
    differencePerTheta = X' * difference;
    grad(1) = 1/m * differencePerTheta(1);
    grad(2:n) = 1/m * ( differencePerTheta(2:n) + lambda * theta(2:n) );
    %grad(2:n) = 1/m * ( differencePerTheta(2:n) );

% =============================================================

end
