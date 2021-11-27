function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));
g = 1./(1.0+exp(-1.0*z));%sigmoid函数 函数里一定要用点除‘./’,因为是矩阵运算，所以要把纬度保持一致。
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).





% =============================================================

end
