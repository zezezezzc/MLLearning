from scipy.stats import ttest_1samp
import numpy as np

print("Null Hypothesis:μ=μ0=30，α=0.05")
ages = [25,36,15,40,28,31,32,30,29,28,27,33,35]
t = (np.mean(ages)-30)/(np.std(ages,ddof=1)/np.sqrt(len(ages)))

ttest,pval = ttest_1samp(ages,30)
print(t,ttest)
if pval < 0.05:
	print("Reject the Null Hypothesis.")
else:
	print("Accept the Null Hypothesis.")


# 独立样本t检验
# 对于第三个问题独立样本t检验，比较两个样本所代表的两个总体均值是否存在显著差异。除了要求样本来自正态分布，还要求两个样本的总体方差相等“方差齐性”。

from scipy.stats import ttest_ind,norm,f
import numpy as np
def ftest(s1,s2):
	'''F检验样本总体方差是否相等'''
	print("Null Hypothesis:var(s1)=var(s2)，α=0.05")
	F = np.var(s1)/np.var(s2)
	v1 = len(s1) - 1
	v2 = len(s2) - 1
	p_val = 1 - 2*abs(0.5-f.cdf(F,v1,v2))
	print(p_val)
	if p_val < 0.05:
		print("Reject the Null Hypothesis.")
		equal_var=False
	else:
		print("Accept the Null Hypothesis.")
	 	equal_var=True
	return equal_var
	 	
def ttest_ind_fun(s1,s2):
	'''t检验独立样本所代表的两个总体均值是否存在差异'''
	equal_var = ftest(s1,s2)
	print("Null Hypothesis:mean(s1)=mean(s2)，α=0.05")
	ttest,pval = ttest_ind(s1,s2,equal_var=equal_var)
	if pval < 0.05:
		print("Reject the Null Hypothesis.")
	else:
		print("Accept the Null Hypothesis.")
	return pval

np.random.seed(42)
s1 = norm.rvs(loc=1,scale=1.0,size=20)
s2 = norm.rvs(loc=1.5,scale=0.5,size=20)
s3 = norm.rvs(loc=1.5,scale=0.5,size=25)

ttest_ind_fun(s1,s2)
ttest_ind_fun(s2,s3)


# KL Divergence
# KL 散度是一种衡量两个概率分布的匹配程度的指标，两个分布差异越大，KL散度越大。注意如果要查看测试集特征是否与训练集相同，
# P代表训练集，Q代表测试集，这个公式对于P和Q并不是对称的



import numpy as np
import scipy.stats

# 随机生成两个离散型分布
x = [np.random.randint(1, 11) for i in range(10)]
print(x)
print(np.sum(x))
px = x / np.sum(x)
print(px)
y = [np.random.randint(1, 11) for i in range(10)]
print(y)
print(np.sum(y))
py = y / np.sum(y)
print(py)

# 利用scipy API进行计算
# scipy计算函数可以处理非归一化情况，因此这里使用
# scipy.stats.entropy(x, y)或scipy.stats.entropy(px, py)均可
KL = scipy.stats.entropy(x, y) 
print(KL)

# 实现
KL = 0.0
for i in range(10):
    KL += px[i] * np.log(px[i] / py[i])
    # print(str(px[i]) + ' ' + str(py[i]) + ' ' + str(px[i] * np.log(px[i] / py[i])))

print(KL)