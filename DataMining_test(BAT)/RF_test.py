#coding:utf-8
#@author:zee(GDUT)
# 工具：jupyter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.decomposition import PCA
import pandas as pd
#读取数据集
data_X= pd.read_csv('D:/test01/features.dat')
labels=pd.read_csv('D:/test01/labels.dat')
print (data_X.info())
X=data_X
y=labels
#格式转换，去掉过多的所有值
c, r = y.shape
y = y.values.reshape(c,)
pca = PCA(n_components = 11)
Xd = pca.fit_transform(X)
clf = RandomForestClassifier(random_state = 14) #随机数种子，控制器控制每次的随机
scores = cross_val_score(clf, Xd , y , scoring='accuracy')
average_accuracy = np.mean(scores) * 100
print("The average accuracy is {0:.1f}%".format(average_accuracy))
#The average accuracy is 94.9%
