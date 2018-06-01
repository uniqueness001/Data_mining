## coding = utf-8 ##
#@author zee(GDUT)
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
data = pd.read_csv('D:/newtrain1.csv')
## 将数据格式转化为float
from sklearn import preprocessing
for f in data.columns:
    if data[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(data[f].values))
        data[f] = lbl.transform(list(data[f].values))

# for f in test.columns:
#     if test[f].dtype=='object':
#         lbl = preprocessing.LabelEncoder()
#         lbl.fit(list(test[f].values))
#         test[f] = lbl.transform(list(test[f].values))

data.fillna((-999), inplace=True)
# test.fillna((-999), inplace=True)

train=np.array(data)
# test=np.array(test)
train = train.astype(float)
# test = test.astype(float)
# x01 = x ( dtype='float')
# sep =int(0.7*len(train ))
# train_data = train [:sep]
# test_data = train[sep:]
y=data['acc_now_delinq']
x=data.drop('acc_now_delinq',axis=1)
X_train,X_test,y_train,y_test=train_test_split(x , y ,test_size=0.1,random_state=0)
columns=X_train.columns
dtrain=xgb.DMatrix(X_train,label=y_train)
dtest=xgb.DMatrix(X_test,label=y_test)
# import pandas as pd
# data = pd.read_csv('D:/newtrain1.csv')
params={
    'booster':'gbtree',
    'objective':'rank:pairwise',
    'eval_metric':'auc',
    'gama':0.1,
    'min_child_weight':2,
    'max_depth':5,
    'lambda':10,
    'subsample':0.7,
    'colsample_bytree':0.7,
    'eta':0.01,
    'tree_method':'exact',
    'seed':0,
    'nthead':7
}

watchlist=[(dtrain,'train'),(dtest,'test')]
model01=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)

# delete_feature=['home_ownership']
# X_train=X_train[[i for i in columns if i not in delete_feature]]
# X_test=X_test[[i for i in columns if i not in delete_feature]]
#
# dtrain=xgb.DMatrix(X_train,label=y_train)
# dtest=xgb.DMatrix(X_test,label=y_test)
# #watchlist=[(dtrain,'train'),(dtest,'test')]
# model02=xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)