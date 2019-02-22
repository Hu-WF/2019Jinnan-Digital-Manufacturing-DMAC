# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold,RepeatedKFold
from scipy import sparse
import re
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import BayesianRidge


def load_data(train_path,test_path):    
    train=pd.read_csv(train_path,encoding='gbk')
    test=pd.read_csv(test_path,encoding='gbk')
    return train,test

def trainTestProcessing(train,test): 
    #删除类别唯一的特征
    for df in [train,test]:
        df.drop(['B3','B13','A13','A18','A23'],axis=1,inplace=True)
    #删除缺失率>90%的特征
    good_cols=list(train.columns)
#    for col in train.columns:
#        rate=train[col].value_counts(normalize=True,dropna=False).values[0]
#        if rate > 0.90:
#            good_cols.remove(col)
#            print('remove:',col,rate)
    good_cols.remove('A2')
    good_cols.remove('B2')
    #剔除收率<0.85的index
    train=train[train['收率']>0.87]
    #剔除收率大于1.00的异常值：
#    train=train[train['收率']<1.00]
    train=train[good_cols]
    good_cols.remove('收率')
    test=test[good_cols]
    #数据清洗-离群值、异常值处理================================================
    #A9
    train['A9']=train['A9'].replace('700','7:00:00')#线下mse下降
    #B14
    train['B14']=train['B14'].replace(40,400)
    test['B14']=test['B14'].replace(785,385)
    #A21
    train=train[train['A21']<=50]
#    train['A21']=train['A21'].replace(train['A21']>50,65)
    #数据清洗-20190119-train、test异常值处理====================================
    #B1
    train['B1']=train['B1'].replace(3.5,320)#190119-新增01
    #B8
#    train=train[train['B8']<=50]#190119-新增02,线下mse上升
##    train['B8']=train['B8'].replace(train['B8']>=50,60)
##    print(train[train['B8']>50])
    #B14
#    train=train[train['B14']>=280]#190119-新增03
##    train['B14']=train['B14'].replace(train['B14']<=260,260)
    #A6
##    train=train[train['A6']>=18]#190119-新增04
    train['A6']=train['A6'].replace(22.7,23)
##    train=train[train['A6']<=80]#190119-新增04
    #A12
##    train=train[train['A12']>=100]#190119-新增05,线下mse较差
##    train=train[train['A12']<=106]#190119-新增06
    test['A12']=test['A12'].replace(104.2,104)#190119-新增07
    #A15
##    train=train[train['A15']>=101]#190119-新增08
##    train=train[train['A15']<=107]#190119-新增09,线下mse较差
    test['A15']=test['A15'].replace(103.9,104)
    #A17
##    train=train[train['A17']>=101]#190119-新增10
##    train['A17']=train['A17'].replace(train['A17']<=101,101)
    train['A17']=train['A17'].replace(101.4,101)#190119-新增11,无明显影响
    train['A17']=train['A17'].replace([103.2,103.5],104)#190119-新增11,无明显影响
    test['A17']=test['A17'].replace(103.6,104)#190119-新增12，几乎没影响
    #A19
##    train=train[train['A19']<=320]#190119-新增13,线下mse值下降较明显
    test['A19']=test['A19'].replace(700,300)#190119-新增14，几乎无影响
##    train['A19']=train['A19'].replace(train['A19']>300,340)
##    test['A19']=test['A19'].replace(test['A19']>300,340)
    #A22
    train['A22']=train['A22'].replace(3.5,9)#190119-新增16，根据排序猜测3.5为9
    #for testB：
    test['A25']=test['A25'].replace(91,71)#
    test['A25']=test['A25'].replace(50,70)#
    test['B1']=test['B1'].replace(316,310)#待定
#    test['A21']=test['A21'].replace(34,40)#待定
    return train,test

def getTime(t):
    try:
        t,m,s=t.split(':')
    except:
        if t == '1900/1/9 7:00':
            return 7*3600/3600
        elif t == '1900/1/1 2:30':
            return (2*3600 + 30*60)/3600
        elif t == -1:
            return -1
        else:
            return 0
    try:
        tm=(int(t)*3600+int(m)*60+int(s))/3600
    except:
        return (30*60)/3600
    return tm

def getDuration(sequence):
    try:
        sh,sm,eh,em=re.findall(r'\d+\.?\d*',sequence)
    except:
        if sequence == -1:
            return -1
    try:
        if int(sh)>int(eh):
            tm=(int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600 + 24
#            print(tm)
            return tm
        else:
            tm=(int(eh)*3600+int(em)*60-int(sm)*60-int(sh)*3600)/3600
            return tm
    except:
        if sequence == '19:-20:05':
            return 1
        if sequence == '16:30-16:30':#针对A20_id1446
            print('exist!')
            return 1
        if sequence == '18:30-15:00':#针对A20_id960
            return 0.5
        if sequence == '18:00-17:00':#针对B4_id1157
            return 1
        if sequence == '22:00-12:00':#针对B11_id609\643
            return 1
        if sequence == '5:00-4:00':#针对B11_id1164
            return 1
        elif sequence == '15:00-1600':
            return 1

def dataProcessing(train,test):
    target=train['收率']
    del train['收率']
    data=pd.concat([train,test],axis=0,ignore_index=True)#ignore_index
    data=data.fillna(-1)
    #train和test缺失值填充
    #A21
    data.loc[data['样本id']=='sample_471','A21'] = 50
    data.loc[data['样本id']=='sample_857','A21'] = 50
    data.loc[data['样本id']=='sample_366','A21'] = 50
    #A24
    data.loc[data['样本id']=='sample_1577','A24'] = '3:00:00'
    #A26
    data.loc[data['样本id']=='sample_1577','A26'] = '3:30:00'
    data.loc[data['样本id']=='sample_534','A26'] = '20:00:00'
    #B1
    data.loc[data['样本id']=='sample_1017','B1'] = 290
    data.loc[data['样本id']=='sample_1439','B1'] = 310
    data.loc[data['样本id']=='sample_601','B1'] = 330
    data.loc[data['样本id']=='sample_584','B1'] = 340
    data.loc[data['样本id']=='sample_392','B1'] = 350
    data.loc[data['样本id']=='sample_373','B1'] = 350
    data.loc[data['样本id']=='sample_337','B1'] = 350
    #B5
    data.loc[data['样本id']=='sample_12','B5'] = '14:00:00'
    #B8
    data.loc[data['样本id']=='sample_122','B8'] = 45
    #B12
    data.loc[data['样本id']=='sample_222','B12'] = 1200
    #convert time to second
    for f in ['A5','A7','A9','A11','A14','A16','A24','A26','B5','B7']:
        data[f]=data[f].apply(getTime)
    #convert sequence to duration
    for f in ['A20','A28','B4','B9','B10','B11']:
        data[f]=data.apply(lambda df:getDuration(df[f]),axis=1)
    data.loc[data['A20']==-1,'A20'] = 0.5#转换后的A20缺失值处理
    #B10、B11、A7、A8缺失值填充：
    data['B10']=data['B10'].replace(-1,0)#
    data['B11']=data['B11'].replace(-1,0)#
    data['A7']=data['A7'].replace(-1,0)#
    data['A8']=data['A8'].replace(-1,0)#
    #A1、A3、A4缺失值填充
    data['A1']=data['A1'].replace(-1,0)#
    data['A3']=data['A3'].replace(-1,0)#
    data['A4']=data['A4'].replace(-1,0)#

    #特征构造-类别型特征-温度&物质/B14
    for f in ['A6','A12','A15','A17','A19','A21','A25','A27','B1','B6','B8',]:
        data[f+'/B14']=data[f]/data['B14']    
        
    #强特-样本id
    data['样本id']=data['样本id'].apply(lambda x:int(x.split('_')[1]))
    data['样本id']=data['样本id'].astype(int)

    categorical_columns=[f for f in data.columns if f not in ['样本id',]]
    numerical_columns=[f for f in data.columns if f not in categorical_columns]
#    print('categorical_columns',categorical_columns,len(categorical_columns))

    #特征构造-数值型特征
    data['B14/A1_A3_A4_A19_B1_B2']=data['B14']/(data['A1']+data['A3']+data['A4']+data['A19']+data['B1']+data['B12'])
    numerical_columns.append('B14/A1_A3_A4_A19_B1_B2')
 
    data['B9/B9_B10_B11']=data['B9']/(data['B9']+data['B10']+data['B11'])
    numerical_columns.append('B9/B9_B10_B11')

    categorical_columns.remove('A1')
    categorical_columns.remove('A3')
    categorical_columns.remove('A4')
    
    #类别特征编码-label encoder
    for f in categorical_columns:
        data[f]=data[f].map(dict(zip(data[f].unique(),range(0,data[f].nunique()))))
    train=data[:train.shape[0]]
    test=data[train.shape[0]:]
    #类别特征-target encoding
    train['target']=target
    train['intTarget']=pd.cut(train['target'],5,labels=False)
    train=pd.get_dummies(train,columns=['intTarget'])
    li= ['intTarget_0.0','intTarget_1.0','intTarget_2.0','intTarget_3.0','intTarget_4.0',]
    mean_columns=[]
    for f1 in categorical_columns:
        cate_rate=train[f1].value_counts(normalize=True,dropna=False).values[0]
        if cate_rate < 0.90:
            for f2 in li:
                col_name='B14_to_'+f1+'_'+f2+'_mean'
                mean_columns.append(col_name)
                order_label=train.groupby([f1])[f2].mean()
                train[col_name]=train['B14'].map(order_label)
                miss_rate=train[col_name].isnull().sum()*100 /train[col_name].shape[0]
                if miss_rate >0:
                    train=train.drop([col_name],axis=1)
                    mean_columns.remove(col_name)
                else:
                    test[col_name]=test['B14'].map(order_label)
    train.drop(li+['target'],axis=1,inplace=True)
  
    #==========================================================================
    data=pd.concat([train,test],axis=0,ignore_index=True)#ignore_index
#    类别特征编码-label encoder
    for f in mean_columns:
        data[f]=data[f].map(dict(zip(data[f].unique(),range(0,data[f].nunique()))))
    train=data[:train.shape[0]]
    test=data[train.shape[0]:]
    #=================================================
#    X_train=train[mean_columns+numerical_columns].values#不包括原始特征
#    X_test=test[mean_columns+numerical_columns].values
    X_train=train[numerical_columns].values
    X_test=test[numerical_columns].values
    #原始特征-onehot encoder
    oh=OneHotEncoder()
    for f in categorical_columns+mean_columns:
        oh.fit(data[f].values.reshape(-1,1))
        X_train=sparse.hstack((X_train,oh.transform(train[f].values.reshape(-1,1))),'csr')
        X_test=sparse.hstack((X_test,oh.transform(test[f].values.reshape(-1,1))),'csr')
    y_train=target.values 
    print('X_train.shape:',X_train.shape)
    print('y_train.shape:',y_train.shape)
    print('X_test.shape:',X_test.shape)
    return X_train,y_train,X_test
        
def lgb_model(X_train,y_train,X_test,):#lightGBM
    params={'num_leaves':120,
           'min_data_in_leaf':20,
           'objective':'regression',
           'max_depth':-1,
           'learning_rate':0.03,#0.03 is better
           'min_child_samples':30,
           'boosting':'gbdt',
           'feature_fraction':1,#调参前为0.9
           'bagging_fraction':0.9,#调参前为0.9，
           'bagging_freq':1,
           'bagging_seed':11,
           'metric':'mse',
           'lambda_l1':0.08,#调参前为0.1
           'lambda_l2':0.02,#调参前为0.01
           'verbosity':-1}
    #5-fold cross-validation
    folds=KFold(n_splits=20,shuffle=True,random_state=2018)
    oof_lgb=np.zeros(X_train.shape[0])
    predictions_lgb=np.zeros(X_test.shape[0])
    for fold , (train_index,valid_index) in enumerate(folds.split(X_train,y_train)):
        print('fold n:{}'.format(fold+1))
        train_data=lgb.Dataset(X_train[train_index],y_train[train_index])
        valid_data=lgb.Dataset(X_train[valid_index],y_train[valid_index])
        num_boost_round=10000
        clf=lgb.train(params,train_data,num_boost_round,
                      valid_sets=[train_data,valid_data],
                      verbose_eval=False,early_stopping_rounds=100,)
        oof_lgb[valid_index]=clf.predict(X_train[valid_index],num_iteration=clf.best_iteration)
        predictions_lgb += clf.predict(X_test,num_iteration=clf.best_iteration)/folds.n_splits 
    print('LGB-CV-MSE-score:{:<0.8f}'.format(mean_squared_error(oof_lgb,y_train)))
    print('0.5*LGB-CV-MSE-score:{:<0.8f}'.format(mean_squared_error(oof_lgb,y_train)*0.5))
    mse=mean_squared_error(oof_lgb,y_train)
    return oof_lgb,predictions_lgb,mse
  
def xgb_model(X_train,y_train,X_test,):
    params={'eta':0.03,#0.03
            'max_depth':10,
            'subsample':1,#初始0.8
            'colsample_bytree':0.9,#初始0.8
            'objective':'reg:linear',
            #'min_child_weight':1,
            'eval_metric':'rmse',
            'silent':True,
            'nthread':4,
            'alpha':0.01}
    folds=KFold(n_splits=20,shuffle=True,random_state=2019)
    oof_xgb=np.zeros(X_train.shape[0])
    predictions_xgb=np.zeros(X_test.shape[0])
    for fold , (train_index,valid_index) in enumerate(folds.split(X_train,y_train)):
        print('fold n:{}'.format(fold+1))
        train_data=xgb.DMatrix(X_train[train_index],y_train[train_index])
        valid_data=xgb.DMatrix(X_train[valid_index],y_train[valid_index])
        watchlist=[(train_data,'train'),(valid_data,'valid_data')]
        clf=xgb.train(dtrain=train_data,num_boost_round=20000,
                      evals=watchlist,early_stopping_rounds=200,
                      verbose_eval=False,params=params)
        oof_xgb[valid_index]=clf.predict(xgb.DMatrix(X_train[valid_index]),ntree_limit=clf.best_ntree_limit)
        predictions_xgb += clf.predict(xgb.DMatrix(X_test),ntree_limit=clf.best_ntree_limit)/folds.n_splits
    mse=mean_squared_error(oof_xgb,y_train)
    print('XGB-CV-MSE-score:{:<0.8f}'.format(mean_squared_error(oof_xgb,y_train)))
    return oof_xgb,predictions_xgb,mse   

def model_stacking(oof_lgb,predictions_lgb,oof_xgb,predictions_xgb,y_train):
    train_stack=np.vstack([oof_lgb,oof_xgb]).transpose()
    test_stack=np.vstack([predictions_lgb,predictions_xgb]).transpose()
    n_splits=5
    n_repeats=2
    folds_stack=RepeatedKFold(n_splits,n_repeats,random_state=2018)#重复n次k折交叉验证
    oof_stack=np.zeros(train_stack.shape[0])
    predictions=np.zeros(test_stack.shape[0])
    for fold , (train_index,valid_index) in enumerate(folds_stack.split(train_stack,y_train)):
        print('fold n:{}'.format(fold+1))
        train_data, train_y=train_stack[train_index],y_train[train_index]
        valid_data, valid_y=train_stack[valid_index],y_train[valid_index]
        clf=BayesianRidge()
        clf.fit(train_data,train_y)
        oof_stack[valid_index]=clf.predict(valid_data)
        predictions += clf.predict(test_stack)/(n_splits*n_repeats)#n_splits * n_repeats
    mse=mean_squared_error(oof_stack,y_train)
    print('Stacking-CV-MSE-score:{:<0.8f}'.format(mean_squared_error(oof_stack,y_train)))
    print('0.5*MSE-score:{:<0.8f}'.format(mean_squared_error(oof_stack,y_train)/2))
    return predictions,mse

def get_submit(test_path,predictions,mse,save_path):
    submit=pd.read_csv(test_path,encoding='gbk',header=None)
    submit=submit.iloc[1:,0:1]
    submit[1]=predictions
    submit[1]=submit[1].apply(lambda x:round(x,3))
    submit.to_csv(save_path,header=None,index=False)
    
    
def run(train_data_path,test_data_path,save_path):
    #load data
    train,test=load_data(train_data_path,test_data_path)
    #data processing
    train,test=trainTestProcessing(train,test)
    X_train,y_train,X_test=dataProcessing(train,test)
    #training and prediction
    oof_lgb , predictions_lgb,mse=lgb_model(X_train,y_train,X_test,)
    oof_xgb , predictions_xgb,mse=xgb_model(X_train,y_train,X_test,)
    predictions,mse=model_stacking(oof_lgb,predictions_lgb,oof_xgb,predictions_xgb,y_train)
    #get submit
    get_submit(test_path=test_data_path,predictions=predictions,mse=mse,save_path=save_path)
    return 0
    
def run_testA_testB_testC():
    train='data/jinnan_round1_train_20181227.csv'
#    test_A='data/jinnan_round1_testA_20181227.csv'
    test_B='data/jinnan_round1_testB_20190121.csv'
    test_C='data/jinnan_round1_test_20190121.csv'
    print('开始训练并测试B榜....')
#    run(train_data_path=train,test_data_path=test_A,save_path='submit_A.csv')
    run(train_data_path=train,test_data_path=test_B,save_path='submit_B.csv')
    print('已生成submit_B.csv')
    print('开始训练并测试C榜....')
    run(train_data_path=train,test_data_path=test_C,save_path='submit_C.csv')
    print('已生成submit_C.csv')
    print('已完成所有操作！')
    return 0
    
if __name__=='__main__':
    run_testA_testB_testC()
    
    
        
        