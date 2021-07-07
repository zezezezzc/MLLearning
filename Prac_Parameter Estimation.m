clc, clear, close all


% @参数估计
A = [1,2,6;
	 7,5,9;
	 0,5,1];

mean(A, 1);

% 标准差
% std(A,a)：a=0时为无偏估计，分母为n-1；a=1时为有偏估计，分母为n
% std(A,a,b)：增加的形参b是维数，若A是二维矩阵，则b=1表示按列分，b=2表示按行分；若为三维以上，b=i就是增多的一维维数
std(A);		% 默认为0
std(A,1);
std(A,0,1);
std(A,0,2);
% 方差 a=0时 分母为n-1；a=1时 分母为n
A = [1,2,5,2,6];
B = [1,3,5;6,3,9;0,0,1];
C = [1,2,5,2,6]';
var(A)
var(A,0)
var(A,1)
var(B)
var(B,0)
var(B,1)
var(C)
var(C,0)
var(C,1)


% 协方差
A = [1,2,5,2,6];
B = [1,3,5;6,3,9;0,0,1];
C = [1,2,5,2,6]';
cov(A)
cov(A,0)
cov(A,1)
cov(B)
cov(B,0)
cov(B,1)

% @PCA
[U, S, V] = svd(covmatrix);

% SVM