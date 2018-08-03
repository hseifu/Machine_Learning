function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
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
    %       of the cost function (computeCostMulti) and gradient here.
    %


    temp = 0;
    temp1 = 0;
    temp2 = 0;
    for i = 1:m
        temp = temp + (theta' * X(i,:)' - y(i));
    end
    
    for j = 1:m
        temp1 = temp1 + ((theta' * X(j,:)' - y(j))*X(j,2));
    end

    for k = 1:m
        temp2 = temp2 + ((theta' * X(k,:)' - y(k))*X(k,3));
    end
    
    theta(1,1) = theta(1,1) - ((alpha/m) * temp);
    theta(2,1) = theta(2,1) - ((alpha/m) * temp1);
    theta(3,1) = theta(3,1) - ((alpha/m) * temp2);    






    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end



end