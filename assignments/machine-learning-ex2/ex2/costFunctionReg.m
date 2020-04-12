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

h_theta = sigmoid(X * theta);
a1 = log(h_theta);
a2 = log(1-h_theta);
J = (-y' * a1 - (1-y)' * a2)/m + lambda/(2*m) * (theta(2:length(theta))' * theta(2:length(theta)));
grad(1) = (X(:,1)' * (h_theta - y)) / m;
for i=2:length(grad)
    grad(i) = X(:,i)' * (h_theta - y)/m + (lambda/m) * theta(i);
end



% =============================================================

end
