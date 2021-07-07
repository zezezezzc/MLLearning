clc, clear, close all

dataSet = load('a');	% [x0, x1, ... , y]	列向量组成
m = 50; 	%样本个数
n = 5;		%参数数量
learnRate = 0.001;

x0 = ones(m, 1);		% 列向量全为1, m*1
% x1 = dataSet(:, 2)
% x2 = dataSet(:, 3)
% ...
% x3 = x1.*x2
% x4 = x1.*x1				% 可能为x1的高次项
% xup_1 = dataSet(1, :)		% 第一组数据
y = dataSet(:, 6);		% m*1

theta = [1;
		 1;
		 1;
		 1;
		 1];				% Parameter initialization, n+1*1
X = [x0, x1, x2, x3, x4];	% Matix m*n+1
X_t = X.';		% n+1*m

decision = (theta.') * X_t;		% 1*m	参数向量转置 × 特征矩阵
% 向量化计算,不能只看维度对应,将累加项转换为矩阵相乘中行和列向量对应相乘相加
temp = ((Sigmoid(decision) - (y.')) * X) / m	% 1*n+1
temp = temp.';		% n+1*1
% 迭代，画出曲线图查看收敛情况
% simutanerous update
for i in itertion
	theta = theta - learnrate * temp;		% gradient descent 凸函数convex func
	err = ;
end

function g = Sigmoid(z)		% Sigmoid Computation function
	g = 1/(1+exp(-z));
end