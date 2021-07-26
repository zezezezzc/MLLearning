clc, clear, close all
% @Logistic Regression train
% dataSet = load('F:\MyGitHub\MLLearning\ex1-ex8-matlab\ex2\ex2data1.txt');
% % [x0, x1, ... , y] 列向量组成
% % x1 = dataSet(:, 2)
% % x2 = dataSet(:, 3)
% % ...
% % x3 = x1.*x2
% % x4 = x1.*x1               % 可能为x1的高次项
% % X = [x0, x1, x2, x3, x4];   
% % X_t = X.';      % n+1*m

% [r, c] = size(dataSet);
% m = r;          % 样本个数
% n = c-1+1;      % 参数数量 -1(标签) +1(偏置分量)

% % Find Indices of Positive and Negative Examples 
% X = dataSet(:, [1, n-1]); % Matix m*n-1
% y = dataSet(:, c);      % m*1
% plotData(X, y);
% % Labels and Legend, Specified in plot order
% xlabel('Exam 1 score'); ylabel('Exam 2 score'); legend('Admitted', 'Not admitted');


% x0 = ones(m, 1);            % x_0 特征列向量全为 1, m*1
% X = [x0, X];

% initial_theta = zeros(n, 1);
% options = optimset('GradObj', 'on', 'MaxIter', 400);
% % options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400);    % 为fmincon求解器创建默认选项
% [theta, cost] = fminunc(@(t)(costLogistic(t, X, y)), initial_theta, options);  % 优化求解器

% % Plot Boundary
% plotDecisionBoundary(theta, X, y); hold on;
% % Labels and Legend Specified in plot order
% xlabel('Exam 1 score'); ylabel('Exam 2 score'); legend('Admitted', 'Not admitted'); hold off;

% prob = sigmoid([1 45 85] * theta);
% fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
%          'probability of %f\n'], prob);
% fprintf('Expected value: 0.775 +/- 0.002\n\n');
 
% p = predict_prob(theta, X);     % Compute accuracy on our training set
% fprintf('Train Accuracy: %f\n', mean(double(p==y)) * 100);
% fprintf('Expected accuracy (approx): 89.0\n');
% fprintf('\n');


% @通过正则化实现
dataSet = load('F:\MyGitHub\MLLearning\ex1-ex8-matlab\ex2\ex2data2.txt');
X = dataSet(:, [1, 2]);
y = dataSet(:, 3);
plotData(X, y); hold on;
xlabel('Microchip Test 1'); ylabel('Microchip Test 2'); legend('y = 1', 'y = 0'); hold off;

X = mapFeature(X(:, 1), X(:, 2));
initial_theta = zeros(size(X, 2), 1);
% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costLogistic_Regularized(initial_theta, X, y, lambda);
fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1); 
% Set regularization parameter lambda
lambda = 0;
% lambda = 1;
% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);
% Optimize
[theta, J, exit_flag] = fminunc(@(t)(costLogistic_Regularized(t, X, y, lambda)), initial_theta, options);
 
% Plot Boundary
plotDecisionBoundary(theta, X, y); hold on; title(sprintf('lambda = %g', lambda)); 
% Labels and Legend
xlabel('Microchip Test 1'); ylabel('Microchip Test 2'); legend('y = 1', 'y = 0', 'Decision boundary'); hold off;
% Compute accuracy on our training set
p = predict_prob(theta, X);
fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');




function plotData(X, y)
    pos = find(y==1);
    neg = find(y == 0);
    figure();       % Plot Examples
    plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7); hold on;
    plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7); hold off;
end

function g = sigmoid(z)		% sigmoid Computation function
	g = zeros(size(z));
    g = 1./(1+exp(-z));
end

function [J, grad] = costLogistic(theta, X, y)
% 求解J(theta)对应的代价函数和梯度值
    m = length(X);
    J = 0;
    grad = zeros(size(theta));
    % Cost function
    X_t = X.';
    decision = (theta.') * X_t;     % 1*m   参数向量转置 × 特征矩阵
    h = sigmoid(decision);      % 向量化计算,不能只看维度对应,将累加项转换为矩阵相乘中行和列向量对应相乘相加
    J = (log(h)*(-y) - log(1-h)*(1-y)) / m;
    % 梯度
    error = h - y.';    % 1*m  all samples error[e_1, e_2, e_3 ... e_m]
    grad = error * X / m;
    grad = grad.';
end

function [J, grad] = costLogistic_Regularized(theta, X, y, lambda)
% 求解J_reg(theta)对应的代价函数和梯度值
    m = length(X);
    J = 0;
    grad = zeros(size(theta));
    % Cost function
    theta2 = theta(2:end);          % m-1*1
    X_t = X.';
    decision = (theta.') * X_t;     % 1*m   参数向量转置 × 特征矩阵
    h = sigmoid(decision);
    J = (log(h)*(-y) - log(1-h)*(1-y))/m + lambda/(2*m) * (theta2.'*theta2);
    % Gradient
    theta(1,1) = 0;     % 不计算偏置分量的梯度
    error = h - y.';    % 1*m
    grad = (error * X).' / m + lambda/m*theta;
end

function plotDecisionBoundary(theta, X, y)
    plotData(X(:, 2:3), y);
    hold on;
    if size(X, 2) <= 3
        % Only need 2 points to define a line, so choose two endpoints
        plot_x = [min(X(:,2))-2, max(X(:,2))+2];

        % Calculate the decision boundary line
        plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

        % Plot, and adjust axes for better viewing
        plot(plot_x, plot_y); legend('Admitted', 'Not admitted', 'Decision Boundary'); axis([30, 100, 30, 100]);
        % Legend, specific for the exercise
    else
        % Here is the grid range
        u = linspace(-1, 1.5, 50);
        v = linspace(-1, 1.5, 50);

        z = zeros(length(u), length(v));
        % Evaluate z = theta*x over the grid
        for i = 1:length(u)
            for j = 1:length(v)
                z(i,j) = mapFeature(u(i), v(j))*theta;
            end
        end
        z = z'; % important to transpose z before calling contour
        % Plot z = 0
        % Notice you need to specify the range [0, 0]
        contour(u, v, z, [0, 0], 'LineWidth', 2)
    end
    hold off
end

function p = predict_prob(theta, X)
    m = size(X, 1);
    p = zeros(m, 1);
    h = sigmoid((theta.') * (X.'));  % predicting the value of probability from the logistic regression algorithm
    h = h.';
    p = round(h);
end

function out = mapFeature(X1, X2)
% 从数据中创建更多类型的特征来更好地拟合
    degree = 6;     % 级数
    out = ones(size(X1(:,1)));      % 添加了一列 x_0
    for i = 1:degree
        for j = 0:i
            out(:, end+1) = (X1.^(i-j)).*(X2.^j);   % 列向量向axis=1方向堆叠
        end
    end
end
