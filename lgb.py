# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 22:07:46 2018

@author: YWZQ
"""

import pandas as pd
from sklearn.model_selection import train_test_split
#import sklearn
import lightgbm as lgb


testb_vid=pd.read_csv(r'.\testb_vid.csv',encoding='gbk')

data=pd.read_csv(r'.\train_x_y.csv',encoding='gbk')
testb=pd.read_csv(r'.\test_x.csv',encoding='gbk')



data.fillna(0,inplace=True)
testb.fillna(0,inplace=True)

#print(data)
#print(data.info())
print(data.isnull().sum().sum())

data_matrix=data.as_matrix()
testa_matrix=testb.as_matrix()

#test_X_matrix=test_X.as_matrix()

X=data_matrix[:,2:]
y=data_matrix[:,1]

testa_x=testa_matrix[:,1:]

#test_X=test_X_matrix[:,1:]
#test_vid=test_X_matrix[:,0]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=16)

lgb_train=lgb.Dataset(X_train,y_train,free_raw_data=False)
lgb_eval=lgb.Dataset(X_test,y_test,reference=lgb_train,free_raw_data=False)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves':63,
    'learning_rate': 0.01,
    'feature_fraction':0.6,
    'bagging_fraction':0.7,
    'bagging_freq':5,
    'verbose': 0,
    'lambda_l1':0.5,
    'lambda_l2':35,
    'min_data_in_leaf':20,
    'min_split_gain':0.1
}
print('model begin:\n')
gbm = lgb.train(params,lgb_train,num_boost_round=2642,verbose_eval=True,early_stopping_rounds=100,valid_sets=[lgb_train,lgb_eval])

print(len(y))
print('y:\n')
y_test_list=list(y)
print(y[2400:2600])

y_predict=gbm.predict(X,num_iteration=gbm.best_iteration) 
print('gbm_best_iteration:\n')
print(gbm.best_iteration)  #一个数值

#print(len(y_predict))
print('y_predict:\n')
print(y_predict)
y_predict_list=list(y_predict)
print(y_predict_list[3900:4100])

y_predict_label=[0 if i<=0.4 else 1 for i in y_predict_list]
num=0
y_len=len(y)
for i in range(y_len):
    if y[i]!=y_predict_label[i]:
        num+=1
print('diffirence:\n')
print(num/y_len)



testa_predict=gbm.predict(testa_x,num_iteration=gbm.best_iteration)
print('len of testb_predict:\n')
print(len(testa_predict))

testa_predict_list=list(testa_predict)
dict_testa={'predict':testa_predict_list}
testa_pd=pd.DataFrame(dict_testa)
testa=pd.concat([testb_vid,testa_pd],axis=1)


testa_predict=testa.sort_values(by='predict',ascending=False)
testa_predict['predict']=testa_predict['predict'].map(lambda x:1 if x>=0.45 else 0 )
print(testa_predict)
#testb_predict.to_csv(r'.\testa_predict_sort.csv',index=None,header=None)
