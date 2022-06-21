import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
cancer=pd.read_excel("/Volumes/Crucial X6/2022课业/大数据案例/乳腺/乳腺.xlsx",sheet_name="乳腺癌26204")
cancert=cancer.T
xt=cancert.drop('Unnamed: 0')
#手肘法
k_range = range(2,20)
k_scores = []
for k in k_range:
    clf = KMeans(n_clusters=k)
    clf.fit(xt)
    scores = clf.inertia_
    k_scores.append(scores)
plt.figure(figsize=(8,6))
plt.plot(k_range, k_scores)
plt.xlabel('k_clusters for Kmeans')
plt.ylabel('inertia')
plt.show()  # 绘制折线图并展示观察
clf = KMeans(n_clusters=5)
clf.fit(xt)
result = clf.predict(xt)
#找出每一类的各个细胞
quantity = pd.Series(clf.labels_).value_counts()
print ("cluster2聚类数量\n", (quantity))
#获取聚类之后每个聚类中心的数据
res0Series = pd.Series(clf.labels_)
res0 = res0Series[res0Series.values == 0]
res1 = res0Series[res0Series.values == 1]
res2 = res0Series[res0Series.values == 2]
res3 = res0Series[res0Series.values == 3]
res4 = res0Series[res0Series.values == 4]
