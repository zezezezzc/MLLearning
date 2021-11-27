from sklearn.cross_validation import train_test_split
import numpy as np

def cal_cov_and_avg(samples):
    """
    给定一个类别的数据，计算协方差矩阵和平均向量
    :param samples: 
    :return: 
    """
    u1 = np.mean(samples, axis=0)
    cov_m = np.zeros((samples.shape[1], samples.shape[1]))
    for s in samples:
        t = s - u1
        cov_m += t * t.reshape(4, 1)
    return cov_m, u1


def fisher(c_1, c_2):
    """
    fisher算法实现(请参考上面推导出来的公式，那个才是精华部分)
    :param c_1: 
    :param c_2: 
    :return: 
    """
    cov_1, u1 = cal_cov_and_avg(c_1)
    cov_2, u2 = cal_cov_and_avg(c_2)
    s_w = cov_1 + cov_2
    u, s, v = np.linalg.svd(s_w)    # 奇异值分解
    s_w_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)
    return np.dot(s_w_inv, u1 - u2)

def judge(sample, w, c_1, c_2):
    """
    true 属于1
    false 属于2
    :param sample:
    :param w:
    :param center_1:
    :param center_2:
    :return:
    """
    u1 = np.mean(c_1, axis=0)
    u2 = np.mean(c_2, axis=0)
    center_1 = np.dot(w.T, u1)
    center_2 = np.dot(w.T, u2)
    pos = np.dot(w.T, sample)
    return abs(pos - center_1) < abs(pos - center_2)

## 数据准备
dna_list = []
with open('dna2','r') as f:
    dna_list = list(map(str.strip,f.readlines()))
    f.close()
print(len(dna_list))

def generate_feature(seq):
    for i in seq:
        size = len(i)
        yield [
        chary.count(i,'a')/size,
        chary.count(i,'t')/size,
        chary.count(i,'c')/size,
        chary.count(i,'g')/size]

X = np.array(list(generate_feature(dna_list)),dtype=float)
y = np.ones(20)
y[10:]=2
X_train, X_test, y_train, y_test = train_test_split(X[:20], y, test_size=0.1)
print(X_train,'\n',y_train)


w = fisher(X_train[:10], X_train[10:20])  # 调用函数，得到参数w
pred = []
for i in range(20):
    pred.append( 1 if judge(X[i], w, X_train[:10], X_train[10:20]) else 2)   # 判断所属的类别
# evaluate accuracy
pred = np.array(pred)
print(y,pred)
print(metrics.accuracy_score(y, pred))
out = []
for i in range(20,40):
    out.append( 1 if judge(X[i], w, X_train[:10], X_train[10:20]) else 2)   # 判断所属的类别
print(out)
