# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 21:04:48 2018

@author: YWZQ
"""

import pandas as pd 
from pandas import merge
import re

testb_voice=pd.read_csv(r'.\voice_test_b.txt','\t',header=None)
train_voice=pd.read_csv(r'.\voice_train.txt','\t',header=None)

testb_voice.columns=['vid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out']
train_voice.columns=['vid','opp_num','opp_head','opp_len','start_time','end_time','call_type','in_out']
train_testb_voice=pd.concat([train_voice,testb_voice])

voice_data=train_testb_voice[['vid']]
#=================================================================
#处理通话起止时间，得到通话时长
#=================================================================
def minus_voice_time(df):
    start_day=int(df['start_time']/1000000)
    start_hour=int(df['start_time']%1000000/10000)
    start_min=int(df['start_time']%10000/100)
    start_second=int(df['start_time']%100)
    
    end_day=int(df['end_time']/1000000)
    end_hour=int(df['end_time']%1000000/10000)
    end_min=int(df['end_time']%10000/100)
    end_second=int(df['end_time']%100)
    
    minus_day=end_day-start_day
    if minus_day>0:
        end_hour+=24
        
    minus_hour=end_hour-start_hour
    if minus_hour>0:
        end_min+=minus_hour*60
        
    minus_min=end_min-start_min
    if minus_min>0:
        end_second+=minus_min*60
    
    time_long=end_second-start_second
  
    df['time_long']=time_long
    
    if int(end_hour)<5:
        a=1
    else:
        a=0
    df['voice_xianshi']=a
    return df
voice_longtime=train_testb_voice.apply(minus_voice_time,axis=1)

#=====================================================================
#存在异常数据，用正则清洗
#=====================================================================
def return_opp_head_class(s):  
    s=str(s)
    p5=r'DDD'
    pattern5=re.compile(p5)
    
    if pattern5.findall(s):
        a=132
    else:
        a=s
    return (a)
voice_longtime['opp_head']=voice_longtime['opp_head'].map(return_opp_head_class)

for i in voice_longtime['opp_head']:
    if i=='DDD':
        i=132
voice_longtime['opp_head']=voice_longtime['opp_head'].apply(pd.to_numeric)
print(voice_longtime.info())
voice_data=merge(voice_data,voice_longtime,on='vid',how='left')

#=======================================================================
#统计每个用户拨打不同号码长度的次数
#=======================================================================
x_voice=voice_longtime
group_opp_len = x_voice['time_long'].groupby([x_voice['vid'],x_voice['opp_len']]).sum()
group_opp_len_unstack=group_opp_len.unstack()
print(group_opp_len_unstack)
print(group_opp_len_unstack.info())
group_opp_len_unstack.to_csv(r'.\voice\group_opp_len.csv')
group_opp_len_unstack=group_opp_len_unstack.reset_index('vid')
voice_data=merge(voice_data,group_opp_len_unstack,on='vid',how='left')


#=================================================================================
#统计每个用户拨打不同号码开头的次数
#=================================================================================
group_opp_head = x_voice['time_long'].groupby([x_voice['vid'],x_voice['opp_head']]).sum()
group_opp_head_unstack=group_opp_head.unstack()
group_opp_head_unstack.fillna(0,inplace=True)
print(group_opp_head_unstack)
print(group_opp_head_unstack.info())
group_opp_head_unstack=group_opp_head_unstack.reset_index('vid')
group_opp_head_unstack.to_csv(r'.\group_opp_head.csv')
voice_data=merge(voice_data,group_opp_head_unstack,on='vid',how='left')

#==============================================================================
#统计不同拨打类型次数
#==============================================================================
x_voice['ci']=1
group_calltype=x_voice['ci'].groupby([x_voice['vid'],x_voice['call_type']]).sum()
group_calltype_unstack=group_calltype.unstack()
group_calltype_unstack.fillna(0,inplace=True)
print(group_calltype_unstack)
print(group_calltype_unstack.isnull().sum().sum())
group_calltype_unstack.to_csv(r'.\group_calltype.csv')
group_calltype_unstack=group_calltype_unstack.reset_index('vid')

voice_data=merge(voice_data,group_calltype_unstack,on='vid',how='left')

#================================================================================
#统计打进打出
#================================================================================
group_in_out = x_voice['ci'].groupby([x_voice['vid'],x_voice['in_out']]).sum()
group_in_out_unstack=group_in_out.unstack()
group_in_out_unstack.fillna(0,inplace=True)
print(group_in_out_unstack)
print(group_in_out_unstack.isnull().sum().sum())
group_in_out_unstack.to_csv(r'.\group_in_out.csv')
group_in_out_unstack=group_in_out_unstack.reset_index('vid')
voice_data=merge(voice_data,group_in_out_unstack,on='vid',how='left')


#============================================================================
#通话时长少于6s次数
#=============================================================================
def longtime_less5s(df):
    df['time_less6s']=0
    if (df['time_long']<6) &(df['in_out']==0):
        df['time_less6s']=1
    return df
voice_longtime_less5s=voice_longtime.apply(longtime_less5s,axis=1)
print(voice_longtime_less5s['time_less6s'].sum())
#print(voice_longtime_less5s['time_long'])

voice_less5s=voice_longtime_less5s['time_less6s'].groupby(voice_longtime_less5s['vid']).sum()
#voice_less5s_pd=pd.DataFrame(voice_less5s,index=range(len(voice_less5s)))
voice_less5s_pd=voice_less5s.to_frame()
voice_less5s_pd=voice_less5s_pd.reset_index()
#voice_less5s_pd.columns=['vid','less5s_ci']
print(voice_less5s_pd)

voice_less5s_pd.to_csv(r'.\group_timelong_less5s.csv',index=None)
voice_data=merge(voice_data,voice_less5s_pd,on='vid',how='left')

#==========================================================================
#打出比例
#=========================================================================

def out_rate(df):
    df['out_rate']=df['0']/(df['0']+df['1'])
    df['pre_label']=0
    if (df['out_rate']>0.8) &((df['0']-df['1'])>100):
        df['pre_label']=1
    return df

voice_out_rate=group_in_out_unstack.apply(out_rate,axis=1)
voice_out_rate.drop(['0','1'],axis=1,inplace=True)
print(voice_out_rate)
print(voice_out_rate['pre_label'].sum())
voice_out_rate.to_csv(r'.\group_inoutrate.csv',index=None)
voice_data=merge(voice_data,voice_out_rate,on='vid',how='left')

#==================
#下面是短信处理
#==================

sms_train=pd.read_csv(r'.\sms_train.txt','\t',header=None)
sms_testb=pd.read_csv(r'.\sms_test_b.txt','\t',header=None)

sms=pd.concat((sms_train,sms_testb))

sms.columns=['vid','opp_num','opp_head','opp_len','start_time','in_out']
sms['ci']=1
print(sms)
data_sms=[['vid']]
group_opp_head=sms['ci'].groupby([sms['vid'],sms['opp_head']]).sum()
group_opp_head_unstack=group_opp_head.unstack()
group_opp_head_unstack.fillna(0,inplace=True)
print(group_opp_head_unstack)
group_opp_head_unstack.to_csv(r'E:\jdata\testb\sms\group_head.csv')
group_opp_head_unstack=group_opp_head_unstack.reset_index('vid')
data_sms=merge(data_sms,group_opp_head_unstack,on='vid',how='left')


group_opp_len=sms['ci'].groupby([sms['vid'],sms['opp_len']]).sum()
group_opp_len_unstack=group_opp_len.unstack()
group_opp_len_unstack.fillna(0,inplace=True)
print(group_opp_len_unstack)
group_opp_len_unstack.to_csv(r'E:\jdata\testb\sms\group_len.csv')
group_opp_len_unstack=group_opp_len_unstack.reset_index('vid')
data_sms=merge(data_sms,group_opp_len_unstack,on='vid',how='left')

group_in_out=sms['ci'].groupby([sms['vid'],sms['in_out']]).sum()
group_in_out_unstack=group_in_out.unstack()
group_in_out_unstack.fillna(0,inplace=True)
print(group_in_out_unstack)
group_in_out_unstack.to_csv(r'.\group_in_out.csv')
group_in_out_unstack=group_in_out_unstack.reset_index('vid')

data_sms=merge(data_sms,group_in_out_unstack,on='vid',how='left')

#=============
#下面是网络
#=============
net_train=pd.read_csv(r'E:\jdata\train\wa_train.txt','\t',header=None)
net_testb=pd.read_csv(r'E:\jdata\testb\testb_data\wa_test_b.txt','\t',header=None)
net=pd.concat((net_train,net_testb))
net.columns=['vid','net_name','vist_times','visit_time_long','up_flow','down_flos','watch_type','date']
#print(net)
data_net=net[['vid']]
group_name = net['vist_times'].groupby([net['vid'],net['net_name']]).sum()
group_name_unstack=group_name.unstack()
group_name_unstack.fillna(0,inplace=True)
#print(group_name_unstack)
#print(group_name_unstack.isnull().sum().sum())
#print(group_name_unstack.info())
group_name_unstack.to_csv(r'.\group_name.csv')
group_name_unstack=group_name_unstack.reset_index('vid')
data_net=merge(data_net,group_name_unstack,on='vid',how='left')


net['ci']=1
#print(net)
group_type=net['ci'].groupby([net['vid'],net['watch_type']]).sum()
net_group_type_unstack=group_type.unstack()
net_group_type_unstack.fillna(0,inplace=True)
print(net_group_type_unstack)
net_group_type_unstack.to_csv(r'.\group_type.csv')
net_group_type_unstack=net_group_type_unstack.reset_index('vid')
data_net=merge(data_net,net_group_type_unstack,on='vid',how='left')
data=merge(voice_data,data_sms,on='vid',how='left')
data=merge(data,data_net,on='vid',how='left')
data.to_csv(r'.\data.csv')
