# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 01:22:21 2020

@author: zezeze
"""
import torch  ## tensor类似于ndarrys
import torch.tensor as tensor
import numpy as np
import time

class nn(object):
	"""docstring for nn"""
	def __init__(self, arg):
		super(nn, self).__init__()
		self.arg = arg
		


#后面的代码直接用to(device)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

a = torch.rand([3,3]).to(device)
# 干其他的活
b = torch.rand([3,3]).to(device)
# 干其他的活
c = torch.rand([3,3]).to(device)

# a = np.array([2,2,2])
# b = torch.from_numpy(a)

# x = torch.rand(5,3)
# print(x)
# x_index = x[:,1]

# print(x_index)  # 索引操作

# x1 = torch.zeros(5, 3, dtype=torch.long)
# print(x1)

# x2 = torch.tensor([5, 3, 1, 3])
# print(x2)

# x3 = x.new_ones(5, 3, dtype=torch.double)
# # new_* methods take in sizes
# print(x)
# x4 = torch.randn_like(x3, dtype=torch.float)
# # override dtype!
# print(x4)
# print(x4.size())    #是一个元组，所以它支持左右的元组操作

# x1.add_(x2)# 任何使张量会发生变化的操作都有一个前缀 ''。例如：x.copy(y), x.t_(), 将会改变 x.

# x = torch.randn(4, 4)
# y = x.view(16)
# z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
# t = x.view(-1, 2)
# print(x.size(), y.size(), z.size(), t.size())

# x = torch.ones(2, 2, requires_grad=True)
# x = tensor([[1., 2.],
#             [3., 4.]], requires_grad=True)      #修改为不同值
# y = x + 2   # y作为操作的结果 被创建，所以它有 grad_fn, 加法结果为AddBackward0
# z = y * y * 3
# out = z.mean()  # mean()即求张量的平均值
# out.backward()  # 等同于out.backward(torch.tensor(1.)) 后向传播，由于 out是一个标量
# # z是个张量，但是根据要求 z.backward() output即z必须是个标量，当然张量也是可以的，所以需
# # 要输入一个大小相同的张量作为参数，这里我们用ones_like函数根据x生成一个张量。因为要对x和y分
# # 别求导数，所以函数z必须是求得的一个值，即标量。然后开始对x，y分别求偏导数。
# # z.backward(torch.ones_like(x))  # o.backward()

# print(x)
# print(out)
# print(x.grad)   # 打印梯度 d(out)/dx

# print(y)
# print(y.grad_fn)
# print(z, out)
# print(x, x.grad)

input = torch.ones([2, 2], requires_grad=False)
w1 = torch.tensor(2.0, requires_grad=True)
w2 = torch.tensor(3.0, requires_grad=True)
w3 = torch.tensor(4.0, requires_grad=True)
## 创建一个简单的计算图
l1 = input * w1
l2 = l1 + w2
l3 = l1 * w3
l4 = l2 * l3
loss = l4.mean()

print(w1.data, w1.grad, w1.grad_fn)
# tensor(2.) None None

print(l1.data, l1.grad, l1.grad_fn)
# tensor([[2., 2.],
#         [2., 2.]]) None <MulBackward0 object at 0x000001EBE79E6AC8>
print(loss.data, loss.grad, loss.grad_fn)
# tensor(40.) None <MeanBackward0 object at 0x000001EBE79D8208>

loss.backward()
print(w1.grad, w2.grad, w3.grad)
# tensor(28.) tensor(8.) tensor(10.)
print(l1.grad, l2.grad, l3.grad, l4.grad, loss.grad)
# None None None None None


x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)

print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():   #以达到暂时不追踪网络参数中的导数的目的,为了减少可能存在的计算和内存消耗
	print((x ** 2).requires_grad)  # False, 同时在对需要求导的叶子张量修改值时可在该语句中修改

 
# a = tensor([[1, 2, 3, 4],
#         [1, 2, 3, 4]]).float()  #norm仅支持floatTensor,a是一个2*4的Tensor
# a0 = torch.norm(a,p=2,dim=0)    #按0维度求2范数
# a1 = torch.norm(a,p=2,dim=1)    #按1维度求2范数
# print(a0)
# print(a1)

# a = torch.rand((2,3,4))
# at = torch.norm(a,p=2,dim=1,keepdim=True)   #保持维度
# af = torch.norm(a,p=2,dim=1,keepdim=False)  #不保持维度
 
# print(a.shape)
# print(at.shape)
# print(af.shape)

if __name__ == '__main__':
	a = torch.randn(10000, 1000)
	b = torch.randn(1000, 2000)

	t0 = time.time()
	c = torch.matmul(a, b)
	t1 = time.time()
	print(a.device, t1-t0, c.norm(2))

	device = torch.device('cuda')	#初始化操作
	a = a.to(device)
	b = b.to(device)
	t0 = time.time()
	c = torch.matmul(a, b)
	t2 = time.time()
	print(a.device, t2-t0, c.norm(2))

	t0 = time.time()
	c = torch.matmul(a, b)
	t2 = time.time()
	print(a.device, t2-t0, c.norm(2))

	#自动求导
	# y = a^2x + bx + c 求对abc求偏导时, x=1的值
	x = torch.tensor(1.)
	a = torch.tensor(1., requires_grad=True)
	b = torch.tensor(2., requires_grad=True)	#需要求导
	c = torch.tensor(3., requires_grad=True)

	y = a**2 *x + b * x + c
	print('before:', a.grad, b.grad, c.grad)
	grads = autograd.grad(y, [a, b, c])
	print('after:', grads[0], grads[1], grads[2])

	#常用的网络层,用他们方便的构建网络
	# nn.Liner
	# nn.Conv2d
	# nn.LSTM		#对时序信号的LSTM层

	# nn.ReLU		#常用激活函数
	# nn.Sigmoid

	# nn.Softmax
	# nn.CrossEntropyLoss
	# nn.MSE		#常用MSE函数
