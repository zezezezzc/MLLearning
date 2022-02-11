# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 22:07:59 2020

@author: 34741
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms  # transforms用于数据预处理

def corr2d(X, K):
'''计算二维互相关运算'''
    h,w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h, j:j+w] * K).sum()
    return Y


def corr2d_multi_in(X,K):
# 多输入通道的互相关计算函数
    return sum(d2l.corr2d(x, k) for x, k in zip(X,K))   # zip中 每次取 X K的最外层维度的小矩阵

X=torch.tensor([[[0.0,1.0,2.0],[3.0,4.0,5.0],[6.0,7.0,8.0]]
    [[1.0,2.0,3.0],[4.0,5.0,6.0],[7.0,8.0,9.0]]])

K=torch.tensor([[[0.0,1.0],[2.0,3.0]],[[1.0, 2.0], [3.0,4.0,5.0]]])
corr2d_multi_in(X,K)

def corr2d_multi_in(X,K):
# 多输入多输出通道的互相关计算函数 X-3d  K-4d
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)   # 在第 0 维度方向中, k为K中取得3d矩阵
K = torch.stack((K, K+1, K+2), 0)
K.shape

def corr2d_multi_in_out_1x1(X, K):
# 1*1的卷积 通过全连接进行实现的
    c_i,h,w=X.shape
    c_o=K.shape[0]
    X=X.reshape((c_i,h*w))
    K=K.reshape((c_o,c_i))
    Y=torch.matmul(K,X)     #矩阵乘法
    return Y.reshape((c_o, h, w))

X=torch.normal(0,1,(3,3,3))
K=torch.normal(0, 1,(2,3,1,1))

Y1=corr2d_multi_in_out_1x1(X,K)
Y2=corr2d_multi_in_out(x,K)
assert float(torch.abs(Y1-Y2).sum())<1e-6 

# torch实现 第一个为输出通道，第二个为输入通道
conv2d =nn.Conv2d(1 1, kernel_size=(3,5), padding=1, stride=2)
comp_conv2d(conv2d, X).shape

class Conv2D (nn.Module):
# 通过继承来构造卷积层
    def init (self, kernel_size):
        super().__init__()
        self.weight =nn. Parameter(torčh.rand (kernel size))
        self.bias = nn.Parameter (torch.zeros (1))

    def forward(self, x):
        return corr2d(x, self,weight) + self.bias

conv2d = Conv2d(1, 1, kernel_size = (1, 2), bias = False)

X = torch.ones((6,8))
X[:, 2:6] = 0

# K = torch.tensor([1.0, -1.0])
K = torch.tensor([[1.0, -1.0]])
Y = corr2d(X, K)
corr2d(X.t(), K)    # 转置后的矩阵

# 通过 X Y 学习 K
conv2d = nn.Conv2d(1,1, kernel_size=(1, 2), bias=False)
X = X.reshape((1, 1, 6, 8))     # reshape 成四维张量,前两维分别是批量大小(batchSize)数和通道数
Y = Y.reshape((1, 1, 6, 7))
for i in range(10):
    Y_hat = conv2d(X)
    l = (Y_hat - Y )**2
    conv2d.zero_grad()
    l.sum().backward()
    conv2d.weight.data[ : ] -= 3e-2 * conv2d.weight.grad
    if (i +1)% 2 == 0:
        print ( f ' batch {i+1}, loss {l.sum ( ) :.3f} ')

conv2d.weight.data.reshape((1, 2))  #结果


class Net(nn.Module):   # 继承父类 Module

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)        
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):   # 前向传播
        # Max pooling over a (2, 2) window  最大值池化层，减少数据量
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()     #参数梯度缓存器置零，用随机的梯度来反向传播
out.backward(torch.randn(1, 10))
