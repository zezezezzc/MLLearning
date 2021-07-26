#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')
# 我们知道，Matplotlib展示的图形允许用户有交互动作，例如缩放、平移、保存等，此时，我们需要调用plt.show()来将程序挂起，
# 直到手动将图像窗口关闭，否则程序与不会向下执行。但将Matplotlib嵌入Jupyter之后，这种Matplotlib生成的图像就处于一种非交互的模式，
# 而%matplotlib inline命令就是激活Matplotlib，为Ipython和Jupyter提供“内嵌后端”支持，也就是作为一个静态图像嵌入Jupyer中，
# 因此Matplotlib就不需要使用plt.show()来主动调用图像展示窗口。

import random
from sklearn import neighbors
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

x1 = np.random.normal(50, 6, 200)
y1 = np.random.normal(5, 0.5, 200)

x2 = np.random.normal(30,6,200)
y2 = np.random.normal(4,0.5,200)

x3 = np.random.normal(45,6,200)
y3 = np.random.normal(2.5, 0.5, 200)


# In[4]:


# 绘图
plt.scatter(x1, y1, c='b', marker='s', s=50, alpha=0.8)
plt.scatter(x2, y2, c='r', marker='^', s=50, alpha=0.8)
plt.scatter(x3, y3, c='g', s=50, alpha=0.8)


# In[26]:


x_val = np.concatenate((x1,x2,x3))
y_val = np.concatenate((y1,y2,y3))
print(type(x_val))

x_diff = max(x_val)-min(x_val)
y_diff = max(y_val)-min(y_val)
xmin = min(x_val)
ymin = min(y_val)

# normalize 极值归一化
x_normalized = [(x-xmin)/(x_diff) for x in x_val]
y_normalized = [(y-ymin)/(y_diff) for y in y_val]
print(type(x_normalized))
print(x_normalized[-1])

# xy_normalized = zip(x_normalized, y_normalized)
xy_normalized = np.stack((x_normalized, y_normalized), axis=-1)
xy = np.stack((x_val, y_val), axis=-1)
print(xy_normalized.shape)# (600, 2)

labels = [1]*200+[2]*200+[3]*200
print(len(labels))


# In[27]:


# 创建clf类
clf = neighbors.KNeighborsClassifier(20)
# 数据学习拟合
clf.fit(xy_normalized, labels)

# 简单测试某几个点 (50,5) 和 (30,3) 两个点附近最近的 5 个样本分别是什么
# nearests = clf.kneighbors([(50/x_diff, 5/y_diff),(30/x_diff, 3/y_diff)], 10, False)
# nearests

# prediction = clf.predict([(50/x_diff, 5/y_diff),(30/x_diff, 3/y_diff)])
# prediction


# In[28]:


x1_test = np.random.normal(50, 6, 100)
y1_test = np.random.normal(5, 0.5, 100)

x2_test = np.random.normal(30,6,100)
y2_test = np.random.normal(4,0.5,100)

x3_test = np.random.normal(45,6,100)
y3_test = np.random.normal(2.5, 0.5, 100)

x_test_val = np.concatenate((x1_test, x2_test, x3_test))#数组拼接
y_test_val = np.concatenate((y1_test, y2_test, y3_test))

x_test_diff = max(x_test_val)-min(x_test_val)
y_test_diff = max(y_test_val)-min(y_test_val)
xmin_test = min(x_test_val)
ymin_test = min(y_test_val)

# normalize score 0.9666666666666667
x_test_normalized = [(x-xmin_test)/(x_test_diff) for x in x_test_val]
y_test_normalized = [(y-ymin_test)/(y_test_diff) for y in y_test_val]

# xy_normalized = zip(x_normalized, y_normalized)
xy_test_normalized = np.stack((x_test_normalized, y_test_normalized), axis=-1)
# 非归一化, 评估得分0.9233333333333333
xy_test = np.stack((x_test_val, y_test_val), axis=-1)

labels_test = [1]*100+[2]*100+[3]*100

score = clf.score(xy_test_normalized, labels_test)
score


# In[33]:


# 分类效果图，只能展示两个维度的数据，首先我们需要生成一个区域里大量的坐标点。这要用到 np.meshgrid() 函数。给定两个 array，
# 比如 x=[1,2,3] 和 y=[4,5]，np.meshgrid(x,y) 会输出两个矩阵

xx,yy = np.meshgrid(np.arange(1,70.1,0.1), np.arange(1,7.01,0.01))

xx_normalized = xx/x_diff
yy_normalized = yy/y_diff

coords = np.c_[xx_normalized.ravel(), yy_normalized.ravel()]# 一个 array 的坐标

Z = clf.predict(coords)
# 当然，Z 是一个一维 array，为了和 xx 还有 yy 相对应，要把Z的形状再转换回矩阵
Z = Z.reshape(xx.shape)
# 用 pcolormesh 画出背景颜色。ListedColormap 是自己生成 colormap 的功能，#rrggbb 颜色的 rgb 代码。pcolormesh 会根据 Z 的值（1、2、3）
# 选择 colormap 里相对应的颜色。pcolormesh 和 ListedColormap 的具体使用方法会在未来关于画图的文章中细讲。

light_rgb = ListedColormap([ '#AAAAFF', '#FFAAAA','#AAFFAA'])
plt.pcolormesh(xx, yy, Z, shading='auto', cmap=light_rgb)
plt.scatter(x1,y1,c='b',marker='s',s=50,alpha=0.8)
plt.scatter(x2,y2,c='r', marker='^', s=50, alpha=0.8)
plt.scatter(x3,y3, c='g', s=50, alpha=0.8)
plt.axis((10, 70,1,7))


# In[31]:


# 概率预测
Z_proba = clf.predict_proba(coords)
# 得到每个坐标点的分类概率值。假设我们想画出红色的概率，那么提取所有坐标的 2 类概率，转换成矩阵形状

Z_proba_reds = Z_proba[:,1].reshape(xx.shape)
# 再选一个预设好的红色调 cmap 画出来
plt.pcolormesh(xx, yy, Z_proba_reds, shading='auto', cmap='Reds')
plt.scatter(x1,y1,c='b',marker='s',s=50,alpha=0.8)
plt.scatter(x2,y2,c='r', marker='^', s=50, alpha=0.8)
plt.scatter(x3,y3, c='g', s=50, alpha=0.8)
plt.axis((10, 70,1,7))

