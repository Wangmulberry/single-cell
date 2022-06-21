#本部分代码为王可欣编写
#将癌细胞数据集聚类后，输出聚类结果，随后抽取100个癌细胞添加到原始数据中


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

#输出聚类结果
#利用Excel整合数据
df0=xt.iloc[res0.index].T
df0.to_excel('/Volumes/Crucial X6/2022课业/大数据案例/r0.xlsx',
                                 index=False, encoding="utf-8")
df1=xt.iloc[res1.index].T
df1.to_excel('/Volumes/Crucial X6/2022课业/大数据案例/r1.xlsx',
                                 index=False, encoding="utf-8")
df2=xt.iloc[res2.index].T
df2.to_excel('/Volumes/Crucial X6/2022课业/大数据案例/r2.xlsx',
                                 index=False, encoding="utf-8")
df3=xt.iloc[res3.index].T
df3.to_excel('/Volumes/Crucial X6/2022课业/大数据案例/r3.xlsx',
                                 index=False, encoding="utf-8")
df4=xt.iloc[res1.index].T
df4.to_excel('/Volumes/Crucial X6/2022课业/大数据案例/r4.xlsx',
                                 index=False, encoding="utf-8")

#读入整合后的数据进行清洗
import csv
data=pd.read_csv("/Volumes/Crucial X6/2022课业/大数据案例/2700+100.csv",index_col=0)
data
data.describe()

#基因筛选
data1=data
index = data1[np.sum(data, axis=1) <3 ].index
data1.drop(index, axis = 0, inplace=True)

#细胞筛选
data2=data1.T
data2
index2 = data2[np.sum(data2, axis=1) <200 ].index
data2.drop(index2, axis = 0, inplace=True)

data_clean=data2.T

#输出清洗后的数据
data_clean.to_csv('/Volumes/Crucial X6/2022课业/大数据案例/data_clean_2722_20623.csv', index=True)
