clc, clear, close all

% 缩放过程可以分为以下几种：缩放到均值为0，方差为1 Standardization——StandardScaler()
% 缩放到0和1之间 Standardization——MinMaxScaler()
% 缩放到-1和1之间 Standardization——MaxAbsScaler()
% 缩放到0和1之间，保留原始数据的分布 Normalization——Normalizer()
% 1就是常说的z-score归一化，2是min-max归一化

% @参数估计 求期望
% X = [X1, X2, ..., Xn]		% n维随机变量
Ex = X.*P/length(X);		% 期望
Rx = (X.'*X) .* P.'*P;		% 相关矩阵
Cx = ((X-Ex).'*(X-Ex)) .* P.'*P;		% 协相关矩阵

% X = [X1, X2, ..., Xn]; Y = [Y1, Y2, ..., Ym]		% n维随机变量
Ex = mean(X, 1);	% 期望
Ey = mean(Y, 1);	% 期望
% Rxy = E( (X.'*Y) );		% 相关矩阵伪代码需要 概率特征函数
% Cxy = E( ((X-Ex).'*(Y-Ey)) );		% 协相关矩阵

% 一维复信号 Z = X+jY	X = [X1, X2, ..., Xn]; Y = [Y1, Y2, ..., Ym]	% 2 维随机变量


% 随机矩阵可拆为行随机向量和列随机向量
% X = [x11 x12 ... x1n;
	   % x21 x22 ... x2n;
	   % ... ... ... ...;
	   % xm1 xm2 ... xmn];
% Vec_c = reshape(X, m*n, 1);	% 列展开
X = X.';
Vec_r = reshape(X, 1, m*n);		% 行展开




% @均值 		mean(x, 1);			% 对期望的无偏估计
A = [1,2,6;
	 7,5,9;
	 0,5,1];
mean(A, 1);
% 标准差
% std(A,a)：a=0时为无偏估计，分母为n-1	a=1时为有偏估计，分母为n
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


% @多维随机变量的协方差
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
