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

h = sigmoid(X*theta); % Set up hypothesis function

thetaMinZero = theta(2:size(theta)); % Taking out first element of vector theta (theta 0)
theta_reg = [0;thetaMinZero]; % Adding the element 0 to front of vector theta so that
                              % our cost function and gradient descent still calculate right

J = (1/m) * (-y'*log(h) - (1-y)' *log(1-h)) + (lambda/(2*m))*theta_reg'*theta_reg; % Passes both tests

%grad = (1/m) * X' * (h - y); % Old Gradient Algorithm
grad = (1/m) * (X' * (h - y ) + lambda*theta_reg);
% =============================================================

end
