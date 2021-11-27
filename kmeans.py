import numpy as np
from scipy.cluster.vq import vq,kmeans,whiten

list1=[88.0,74.0,96.0,85.0]
list2=[92.0,99.0,95.0,94.0]
list3=[91.0,87.0,99.0,95.0]
list4=[78.0,99.0,97.0,81.0]
list5=[88.0,78.0,98.0,84.0]
list6=[100.0,95.0,100.0,92.0]
data=np.array([list1, list2, list3, list4, list5, list6])   # 将一个列表转为np数组
print(data)
whiten=whiten(data) # 算出各列元素的标准差，形成一个新的数组
centroids,_=kmeans(whiten,2)    # 分为两类
result,_=vq(whiten,centroids)
print(result)

# import numpy as np
# from sklearn.cluster import KMeans
# list1=[88.0,74.0,96.0,85.0]
# list2=[92.0,99.0,95.0,94.0]
# list3=[91.0,87.0,99.0,95.0]
# list4=[78.0,99.0,97.0,81.0]
# list5=[88.0,78.0,98.0,84.0]
# list6=[100.0,95.0,100.0,92.0]
# X=np.array([list1, list2, list3, list4, list5, list6])
# kmeans=KMeans(n_clusters=2).fit(X)
# pred=kmeans.predict(X)
# print(pred)
