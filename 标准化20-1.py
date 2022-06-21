#!/usr/bin/env python
# coding: utf-8

# # 导入标准化数据

# In[348]:


import pandas as pd
d1 = pd.read_csv('D:\\A-czy&lyy\\大数据案例分析\\标准化20-1.csv')
d1.head(6)


# In[349]:


d1 = d1.drop(['genes'],axis=1)#去除第一列


# # 主成分降维

# In[350]:


from sklearn.decomposition import PCA
import numpy as np 
pca = PCA(n_components=50)   
pca.fit(d1.T)                  
x1=pca.fit_transform(d1.T) 


# In[352]:


#降维后的结果保存为数据框格式
x1_1=pd.DataFrame(x1)
#将数据转换为数组形式
x1_3=np.array(x1_1)


# # K-means聚类

# In[355]:


#各种距离定义以及k-means聚类定义
from numpy import *
XT=x1_3.T
S=np.cov(XT)
SI = np.linalg.inv(S)#协方差的逆

def distance_o(vector1, vector2):
    """计算欧氏距离"""
    return sqrt(sum(power(vector1-vector2, 2))) 

def distance_m(x,y):
    """计算马氏距离"""
    delta=x-y
    d=np.sqrt(np.dot(np.dot(delta,SI),delta.T))
    return d

def distance_y(a, b):
    """计算余弦相似度"""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    cos = np.dot(a,b)/(a_norm * b_norm)
    return cos # 返回两个向量的距离

def rand_center(dataSet, k):
    """构建一个包含K个随机质心的集合"""
    n = shape(dataSet)[1]  # 获取样本特征值


    # 初始化质心，创建(k,n)个以0填充的矩阵
    centroids = mat(zeros((k, n)))  # 每个质心有n个坐标值，总共要k个质心
    # 遍历特征值
    for j in range(n):
        # 计算每一列的最小值
        minJ = min(dataSet[:, j])
        # 计算每一列的范围值
        rangeJ = float(max(dataSet[:, j]) - minJ)
        # 计算每一列的质心，并将其赋给centroids
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids   # 返回质心


def k_means_o(dataSet,k,distMeas = distance_o,creatCent = rand_center):
    """K-means聚类算法"""
    m = shape(dataSet)[0] # 行数
    # 建立簇分配结果矩阵，第一列存放该数据所属中心点，第二列是该数据到中心点的距离
    clusterAssment = mat(zeros((m, 2)))
    centroids = creatCent(dataSet, k) # 质心，即聚类点
    # 用来判定聚类是否收敛
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 把每一个数据划分到离他最近的中心点
            minDist = inf # 无穷大
            minIndex = -1 #初始化
            for j in range(k):
                # 计算各点与新的聚类中心的距离
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI < minDist:
                    # 如果第i个数据点到第j中心点更近，则将i归属为j
                    minDist = distJI
                    minIndex = j
            # 如果分配发生变化，则需要继续迭代
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            # 并将第i个数据点的分配情况存入字典
            clusterAssment[i,:] = minIndex,minDist**2
       # print(centroids)
        for cent in range(k):  # 重新计算中心点
            # 去第一列等于cent的所有列
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 算出这些数据的中心点
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def k_means_m(dataSet,k,distMeas = distance_m,creatCent = rand_center):
    """K-means聚类算法"""
    m = shape(dataSet)[0] # 行数
    # 建立簇分配结果矩阵，第一列存放该数据所属中心点，第二列是该数据到中心点的距离
    clusterAssment = mat(zeros((m, 2)))
    centroids = creatCent(dataSet, k) # 质心，即聚类点
    # 用来判定聚类是否收敛
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 把每一个数据划分到离他最近的中心点
            minDist = inf # 无穷大
            minIndex = -1 #初始化
            for j in range(k):
                # 计算各点与新的聚类中心的距离
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI< minDist:
                    # 如果第i个数据点到第j中心点更近，则将i归属为j
                    minDist = distJI
                    minIndex = j
            # 如果分配发生变化，则需要继续迭代
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            # 并将第i个数据点的分配情况存入字典
            clusterAssment[i,:] = minIndex,minDist**2
        #print(centroids)
        for cent in range(k):  # 重新计算中心点
            # 去第一列等于cent的所有列
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 算出这些数据的中心点
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment

def k_means_y(dataSet,k,distMeas = distance_y,creatCent = rand_center):
    """K-means聚类算法"""
    m = shape(dataSet)[0] # 行数
    # 建立簇分配结果矩阵，第一列存放该数据所属中心点，第二列是该数据到中心点的距离
    clusterAssment = mat(zeros((m, 2)))
    centroids = creatCent(dataSet, k) # 质心，即聚类点
    # 用来判定聚类是否收敛
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 把每一个数据划分到离他最近的中心点
            maxDist = -1 # 无穷小
            maxIndex = -1 #初始化
            for j in range(k):
                # 计算各点与新的聚类中心的距离
                distJI = distMeas(centroids[j,:],dataSet[i,:])
                if distJI > maxDist:
                    # 如果第i个数据点到第j中心点更近，则将i归属为j
                    maxDist = distJI
                    maxIndex = j
            # 如果分配发生变化，则需要继续迭代
            if clusterAssment[i,0] != maxIndex:
                clusterChanged = True
            # 并将第i个数据点的分配情况存入字典
            clusterAssment[i,:] = maxIndex,maxDist**2
        #print(centroids)
        for cent in range(k):  # 重新计算中心点
            # 去第一列等于cent的所有列
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 算出这些数据的中心点
            centroids[cent, :] = mean(ptsInClust, axis=0)
    return centroids, clusterAssment


# In[207]:


#欧氏距离轮廓系数法
def distCal(vector1, vector2):
    return sqrt(sum(power(vector1-vector2, 2))) 

if __name__ == '__main__':
    data = x1_3
    m = np.shape(data)[0]  # 一共有m行数据

    for k in range(2, 10):  # 簇的个数取值在2到9之间
    
        clusterAssment = k_means_o(data, k)[1]  # 进行二分类，返回保存簇索引的矩阵
        s_sum = 0  # 所有簇的s值
        cluster_s = 0  # 一个簇所有点的的s值
        
        for cent in range(k):  # 对于每一个簇
            category = np.nonzero(clusterAssment[:, 0] == cent)[0]  # 得到簇索引为cent的值的位置，形式类似为[1, 4 ,6]
            clusterNum = len(category)  # 该簇中点的个数
            s = 0
            for index, lineNum in enumerate(category):  # 对于簇中的每一个点，index为索引，lineNum为点所在的行数

                # 计算该点到簇内其他点的距离之和的平均值
                innerSum = 0
                for i in range(clusterNum):
                    if i == index:
                        continue  # 若为当前该点，则跳出本次循环
                    dis = distCal(data[category[i]], data[lineNum])  # 若二者为不同点，计算二者之间的距离
                    innerSum += dis  # 将之保存到内部距离
                a = innerSum / clusterNum

                # 计算该点到其他簇所有点距离之和的最小平均值
                minDis = np.inf  # 设定初始最小值为无穷大
                for other_cent in range(k):  # 对于每一个簇
                    if other_cent != cent:  # 如果和上面给定的簇不一样
                        other_category = np.nonzero(clusterAssment[:, 0] == other_cent)[0]  # 得到簇里面点对应的行数
                        other_clusterNum = len(other_category)  # 该簇中点的个数
                        other_sum = 0
                        for other_lineNum in other_category:  # 对于簇中的每一个点
                            other_dis = distCal(data[other_lineNum], data[lineNum])
                            other_sum += other_dis
                        if other_clusterNum!=0:
                            other_sum = other_sum / other_clusterNum  # 求平均
                        else:
                            other_sum =0
                        if other_sum < minDis:
                            minDis = other_sum  # 如果一个点距离另外一个簇所有点的距离小于当前最小值，则更新
                b = minDis

                s += (b-a) / max(a, b)  # 每一个点的轮廓系数
            cluster_s += s  # 每一个簇的s值
        s_sum = cluster_s / m  # 取平均
        
        print("当前k的值为：%d" % k)
        print("轮廓系数为：%s" % str(s_sum))
        print('***' * 20)


# In[ ]:


#马氏距离轮廓系数法
def distCal(vector1, vector2):
    return sqrt(sum(power(vector1-vector2, 2))) 

if __name__ == '__main__':
    data = x1_3
    m = np.shape(data)[0]  # 一共有m行数据

    for k in range(2, 10):  # 簇的个数取值在2到9之间
    
        clusterAssment = k_means_m(data, k)[1]  # 进行二分类，返回保存簇索引的矩阵
        s_sum = 0  # 所有簇的s值
        cluster_s = 0  # 一个簇所有点的的s值
        
        for cent in range(k):  # 对于每一个簇
            category = np.nonzero(clusterAssment[:, 0] == cent)[0]  # 得到簇索引为cent的值的位置，形式类似为[1, 4 ,6]
            clusterNum = len(category)  # 该簇中点的个数
            s = 0
            for index, lineNum in enumerate(category):  # 对于簇中的每一个点，index为索引，lineNum为点所在的行数

                # 计算该点到簇内其他点的距离之和的平均值
                innerSum = 0
                for i in range(clusterNum):
                    if i == index:
                        continue  # 若为当前该点，则跳出本次循环
                    dis = distCal(data[category[i]], data[lineNum])  # 若二者为不同点，计算二者之间的距离
                    innerSum += dis  # 将之保存到内部距离
                a = innerSum / clusterNum

                # 计算该点到其他簇所有点距离之和的最小平均值
                minDis = np.inf  # 设定初始最小值为无穷大
                for other_cent in range(k):  # 对于每一个簇
                    if other_cent != cent:  # 如果和上面给定的簇不一样
                        other_category = np.nonzero(clusterAssment[:, 0] == other_cent)[0]  # 得到簇里面点对应的行数
                        other_clusterNum = len(other_category)  # 该簇中点的个数
                        other_sum = 0
                        for other_lineNum in other_category:  # 对于簇中的每一个点
                            other_dis = distCal(data[other_lineNum], data[lineNum])
                            other_sum += other_dis
                        if other_clusterNum!=0:
                            other_sum = other_sum / other_clusterNum  # 求平均
                        else:
                            other_sum =0
                        if other_sum < minDis:
                            minDis = other_sum  # 如果一个点距离另外一个簇所有点的距离小于当前最小值，则更新
                b = minDis

                s += (b-a) / max(a, b)  # 每一个点的轮廓系数
            cluster_s += s  # 每一个簇的s值
        s_sum = cluster_s / m  # 取平均
        
        print("当前k的值为：%d" % k)
        print("轮廓系数为：%s" % str(s_sum))
        print('***' * 20)


# In[208]:


#余弦距离轮廓系数法
def distCal(vector1, vector2):
    return sqrt(sum(power(vector1-vector2, 2))) 

if __name__ == '__main__':
    data = x1_3
    m = np.shape(data)[0]  # 一共有m行数据

    for k in range(2, 10):  # 簇的个数取值在2到9之间
    
        clusterAssment = k_means_y(data, k)[1]  # 进行二分类，返回保存簇索引的矩阵
        s_sum = 0  # 所有簇的s值
        cluster_s = 0  # 一个簇所有点的的s值
        
        for cent in range(k):  # 对于每一个簇
            category = np.nonzero(clusterAssment[:, 0] == cent)[0]  # 得到簇索引为cent的值的位置，形式类似为[1, 4 ,6]
            clusterNum = len(category)  # 该簇中点的个数
            s = 0
            for index, lineNum in enumerate(category):  # 对于簇中的每一个点，index为索引，lineNum为点所在的行数

                # 计算该点到簇内其他点的距离之和的平均值
                innerSum = 0
                for i in range(clusterNum):
                    if i == index:
                        continue  # 若为当前该点，则跳出本次循环
                    dis = distCal(data[category[i]], data[lineNum])  # 若二者为不同点，计算二者之间的距离
                    innerSum += dis  # 将之保存到内部距离
                a = innerSum / clusterNum

                # 计算该点到其他簇所有点距离之和的最小平均值
                minDis = np.inf  # 设定初始最小值为无穷大
                for other_cent in range(k):  # 对于每一个簇
                    if other_cent != cent:  # 如果和上面给定的簇不一样
                        other_category = np.nonzero(clusterAssment[:, 0] == other_cent)[0]  # 得到簇里面点对应的行数
                        other_clusterNum = len(other_category)  # 该簇中点的个数
                        other_sum = 0
                        for other_lineNum in other_category:  # 对于簇中的每一个点
                            other_dis = distCal(data[other_lineNum], data[lineNum])
                            other_sum += other_dis
                        if other_clusterNum!=0:
                            other_sum = other_sum / other_clusterNum  # 求平均
                        else:
                            other_sum =0
                        if other_sum < minDis:
                            minDis = other_sum  # 如果一个点距离另外一个簇所有点的距离小于当前最小值，则更新
                b = minDis

                s += (b-a) / max(a, b)  # 每一个点的轮廓系数
            cluster_s += s  # 每一个簇的s值
        s_sum = cluster_s / m  # 取平均
        
        print("当前k的值为：%d" % k)
        print("轮廓系数为：%s" % str(s_sum))
        print('***' * 20)


# In[209]:


#k值确定后的三种聚类的轮廓系数
if __name__ == '__main__':
    data = x1_3
    m = np.shape(data)[0]  # 一共有m行数据
    
    clusterAssment_o = k_means_o(data, 2)[1]  # 进行二分类，返回保存簇索引的矩阵
    clusterAssment_m = k_means_m(data, 2)[1] 
    clusterAssment_y = k_means_y(data, 2)[1] 
    s_sum_o = 0  # 所有簇的s值
    s_sum_m = 0 
    s_sum_y = 0 
    cluster_s_o = 0  # 一个簇所有点的的s值
    cluster_s_m = 0
    cluster_s_y = 0
        
    for cent in range(2):  # 对于每一个簇
        category_o = np.nonzero(clusterAssment_o[:, 0] == cent)[0]  # 得到簇索引为cent的值的位置，形式类似为[1, 4 ,6]
        category_m = np.nonzero(clusterAssment_m[:, 0] == cent)[0]
        category_y = np.nonzero(clusterAssment_y[:, 0] == cent)[0]
        clusterNum_o = len(category_o)  # 该簇中点的个数
        clusterNum_m = len(category_m) 
        clusterNum_y = len(category_y) 
        s = 0
        for index, lineNum in enumerate(category_o):  # 对于簇中的每一个点，index为索引，lineNum为点所在的行数

            # 计算该点到簇内其他点的距离之和的平均值
            innerSum = 0
            for i in range(clusterNum_o):
                if i == index:
                    continue  # 若为当前该点，则跳出本次循环
                dis = distCal(data[category_o[i]], data[lineNum])  # 若二者为不同点，计算二者之间的距离
                innerSum += dis  # 将之保存到内部距离
            a= innerSum / clusterNum_o

            # 计算该点到其他簇所有点距离之和的最小平均值
            minDis = np.inf  # 设定初始最小值为无穷大
            for other_cent in range(2):  # 对于每一个簇
                if other_cent != cent:  # 如果和上面给定的簇不一样
                    other_category = np.nonzero(clusterAssment_o[:, 0] == other_cent)[0]  # 得到簇里面点对应的行数
                    other_clusterNum = len(other_category)  # 该簇中点的个数
                    other_sum = 0
                    for other_lineNum in other_category:  # 对于簇中的每一个点
                        other_dis = distCal(data[other_lineNum], data[lineNum])
                        other_sum += other_dis
                    if other_clusterNum!=0:
                        other_sum = other_sum / other_clusterNum  # 求平均
                    else:
                        other_sum =0
                    if other_sum < minDis:
                        minDis = other_sum  # 如果一个点距离另外一个簇所有点的距离小于当前最小值，则更新
            b = minDis

            s += (b-a) / max(a, b)  # 每一个点的轮廓系数
        cluster_s_o += s  # 每一个簇的s值
        s_sum_o = cluster_s_o / m  # 取平均
    
        for index, lineNum in enumerate(category_m):  # 对于簇中的每一个点，index为索引，lineNum为点所在的行数

            # 计算该点到簇内其他点的距离之和的平均值
            innerSum = 0
            for i in range(clusterNum_m):
                if i == index:
                    continue  # 若为当前该点，则跳出本次循环
                dis = distCal(data[category_m[i]], data[lineNum])  # 若二者为不同点，计算二者之间的距离
                innerSum += dis  # 将之保存到内部距离
            a= innerSum / clusterNum_m

            # 计算该点到其他簇所有点距离之和的最小平均值
            minDis = np.inf  # 设定初始最小值为无穷大
            for other_cent in range(2):  # 对于每一个簇
                if other_cent != cent:  # 如果和上面给定的簇不一样
                    other_category = np.nonzero(clusterAssment_m[:, 0] == other_cent)[0]  # 得到簇里面点对应的行数
                    other_clusterNum = len(other_category)  # 该簇中点的个数
                    other_sum = 0
                    for other_lineNum in other_category:  # 对于簇中的每一个点
                        other_dis = distCal(data[other_lineNum], data[lineNum])
                        other_sum += other_dis
                    if other_clusterNum!=0:
                        other_sum = other_sum / other_clusterNum  # 求平均
                    else:
                        other_sum =0
                    if other_sum < minDis:
                        minDis = other_sum  # 如果一个点距离另外一个簇所有点的距离小于当前最小值，则更新
            b = minDis

            s += (b-a) / max(a, b)  # 每一个点的轮廓系数
        cluster_s_m += s  # 每一个簇的s值
        s_sum_m = cluster_s_m / m  # 取平均
        
        
        for index, lineNum in enumerate(category_y):  # 对于簇中的每一个点，index为索引，lineNum为点所在的行数

            # 计算该点到簇内其他点的距离之和的平均值
            innerSum = 0
            for i in range(clusterNum_y):
                if i == index:
                    continue  # 若为当前该点，则跳出本次循环
                dis = distCal(data[category_y[i]], data[lineNum])  # 若二者为不同点，计算二者之间的距离
                innerSum += dis  # 将之保存到内部距离
            a= innerSum / clusterNum_y

            # 计算该点到其他簇所有点距离之和的最小平均值
            minDis = np.inf  # 设定初始最小值为无穷大
            for other_cent in range(2):  # 对于每一个簇
                if other_cent != cent:  # 如果和上面给定的簇不一样
                    other_category = np.nonzero(clusterAssment_y[:, 0] == other_cent)[0]  # 得到簇里面点对应的行数
                    other_clusterNum = len(other_category)  # 该簇中点的个数
                    other_sum = 0
                    for other_lineNum in other_category:  # 对于簇中的每一个点
                        other_dis = distCal(data[other_lineNum], data[lineNum])
                        other_sum += other_dis
                    if other_clusterNum!=0:
                        other_sum = other_sum / other_clusterNum  # 求平均
                    else:
                        other_sum =0
                    if other_sum < minDis:
                        minDis = other_sum  # 如果一个点距离另外一个簇所有点的距离小于当前最小值，则更新
            b = minDis

            s += (b-a) / max(a, b)  # 每一个点的轮廓系数
        cluster_s_y += s  # 每一个簇的s值
        s_sum_y = cluster_s_y / m  # 取平均    
    
    
    
    print("欧氏距离轮廓系数为：%s" % str(s_sum_o))
    print("马氏距离轮廓系数为：%s" % str(s_sum_m))
    print("余弦距离轮廓系数为：%s" % str(s_sum_y))


# In[356]:


#马氏距离聚类
myCentroids_m, clusterAssing_m = k_means_m(x1_3,2)


# In[357]:


#聚类类别
c_m=array(clusterAssing_m)[:, 0].T


# In[358]:


#可视化
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(15, 8), dpi=200)
sc=plt.scatter(x1_3[c_m==0][:, 0], x1_3[c_m==0][:, 1], c='deepskyblue' ,  label='0')
sc=plt.scatter(x1_3[c_m==1][:, 0], x1_3[c_m==1][:, 1], c= 'black', label='1')
plt.colorbar(sc)
plt.title('pca',fontsize=24)
plt.legend()
plt.show()


# In[359]:


class_final= pd.DataFrame(clusterAssing_m)
new_col=['class','dis']
class_final.columns=new_col
class_final


# In[360]:


d3=d1.T
d3.head(6)


# In[362]:


#癌细胞正确聚类的个数
t=0
for i in range(2622,2642):
    if class_final['class'][i]==1.0:
        t=t+1
t


# In[363]:


#与癌细胞聚为一类的个数
t1=0
a=[]
for i in range(len(class_final['class'])):
    if class_final['class'][i]==1.0:
        t1=t1+1
        a.append(d3.index[i])
t1


# In[366]:


#保存数据
a1=pd.DataFrame(a)
a1.to_csv("D:\\A-czy&lyy\\大数据案例分析\\终结版1.csv",index=False,sep=',')


# In[ ]:




