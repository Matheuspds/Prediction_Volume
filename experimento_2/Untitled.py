
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from datetime import time, timedelta
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import math
import random
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.cross_validation import ShuffleSplit, train_test_split
from sklearn.metrics import mean_squared_error
path_final = 'dados_finais/'
from sklearn import preprocessing
from sklearn import svm
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df_test1 = pd.read_csv(path_final+"test1.csv")
df_test2 = pd.read_csv(path_final+"test2.csv")
df_test3 = pd.read_csv(path_final+"test3.csv")
df_test4 = pd.read_csv(path_final+"test4.csv")
df_train1 = pd.read_csv(path_final+"train1.csv")
df_train2 = pd.read_csv(path_final+"train2.csv")
df_train3 = pd.read_csv(path_final+"train3.csv")
df_train4 = pd.read_csv(path_final+"train4.csv")


# In[47]:


df_teste_para_weekday = pd.read_csv('data_after_process/teste_final_para_weekday.csv')


# In[48]:


df_train1 = df_train1.rename(columns = {'time_start':'time', 'weekday': 'week'})
df_train2 = df_train2.rename(columns = {'time_start':'time', 'weekday': 'week'})
df_train3 = df_train3.rename(columns = {'time_start':'time', 'weekday': 'week'})
df_train4 = df_train4.rename(columns = {'time_start':'time', 'weekday': 'week'})
df_test1 = df_test1.rename(columns = {'time_start':'time', 'weekday': 'week'})


# In[49]:


#removendo outliers do df_train1
df_remove = df_train1.loc[(df_train1['day'] >= 1) & (df_train1['day'] <= 7) ]

df_train1 = df_train1.drop(df_remove.index)


# In[50]:


#removendo outliers do df_train2
df_remove = df_train2.loc[(df_train2['day'] >= 1) & (df_train2['day'] <= 7) ]

df_train2 = df_train2.drop(df_remove.index)


# In[51]:


#removendo outliers do df_train3
df_remove = df_train3.loc[(df_train3['day'] >= 1) & (df_train3['day'] <= 7) ]

df_train3 = df_train3.drop(df_remove.index)


# In[52]:


#removendo outliers do df_train4
df_remove = df_train4.loc[(df_train4['day'] >= 1) & (df_train4['day'] <= 7) ]

df_train4 = df_train4.drop(df_remove.index)


# In[53]:


def adiciona_media_desvio_por_dia(df):
    df['media_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])["volume"].transform(np.mean)
    df['desvio_padrao_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])["volume"].transform(np.std)
    df['desvio_padrao_hora_dia'].fillna(df.groupby(['direction', 'tollgate_id', 'day', 'hour'])["volume"].transform(np.mean), inplace=True)
    df['min_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])['volume'].transform(np.min)
    df['max_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])['volume'].transform(np.max)
    df['mediana_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])['volume'].transform(np.median)
    return df


# In[54]:


def medidas_volume_tollgate_direction_am_pm(df):
    df['min_volume_dia_am_pm'] = df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.min)
    df['max_volume_dia_am_pm'] = df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.max)
    df['mediana_volume_dia_am_pm'] = df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.median)
    df['media_volume_dia_am_pm'] = df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.mean)
    df['desvio_padrao_am_pm'] = df.groupby(['day','direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.std)
    df['desvio_padrao_am_pm'].fillna(df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.mean), inplace=True)
    return df


# In[55]:


del df_train1['media_volume']
del df_train1['desvio_padrao']

del df_train2['media_volume']
del df_train2['desvio_padrao']

del df_train3['media_volume']
del df_train3['desvio_padrao']

del df_train4['media_volume']
del df_train4['desvio_padrao']

del df_test1['media_volume']
del df_test1['desvio_padrao']


# In[56]:


df_train1 =adiciona_media_desvio_por_dia(df_train1)
df_train2 = adiciona_media_desvio_por_dia(df_train2)
df_train3 = adiciona_media_desvio_por_dia(df_train3)
df_train4 = adiciona_media_desvio_por_dia(df_train4)
df_test1 = adiciona_media_desvio_por_dia(df_test1)


# In[57]:


df_train1 = medidas_volume_tollgate_direction_am_pm(df_train1)
df_train2 = medidas_volume_tollgate_direction_am_pm(df_train2)
df_train3 = medidas_volume_tollgate_direction_am_pm(df_train3)
df_train4 = medidas_volume_tollgate_direction_am_pm(df_train4)
df_test1 = medidas_volume_tollgate_direction_am_pm(df_test1)


# In[58]:


def adiciona_media_desvio_por_janela_dia_semana_1(df):
    df['time'] = pd.to_datetime(df['time'], format = '%Y-%m-%d %H:%M:%S')
    df['date'] = df['time'].dt.date
    mask = (df['time'] > '2016-09-18 00:00:00') & (df['time'] < '2016-09-26 00:00:00')
    df = df.loc[mask]
    df['min_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.min)
    df['max_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.max)
    df['mediana_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.median)
    df['media_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean)
    df['desvio_padrao_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.std)
    df['desvio_padrao_weekday'].fillna(df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
    return df

def adiciona_media_desvio_por_janela_dia_semana_2(df):
    
    df['time'] = pd.to_datetime(df['time'], format = '%Y-%m-%d %H:%M:%S')
    df['date'] = df['time'].dt.date
    mask2 = (df['time'] > '2016-09-18 00:00:00') & (df['time'] < '2016-10-10 00:00:00')
    df = df.loc[mask2]
    df['min_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.min)
    df['max_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.max)
    df['mediana_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.median)
    df['media_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean)
    df['desvio_padrao_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.std)
    df['desvio_padrao_weekday'].fillna(df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
    
    return df

def adiciona_media_desvio_por_janela_dia_semana_3(df):
    
    df['time'] = pd.to_datetime(df['time'], format = '%Y-%m-%d %H:%M:%S')
    df['date'] = df['time'].dt.date
    mask3 = (df['time'] > '2016-09-18 00:00:00') & (df['time'] < '2016-10-17 00:00:00')
    df = df.loc[mask3]
    df['min_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.min)
    df['max_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.max)
    df['mediana_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.median)
    df['media_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean)
    df['desvio_padrao_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.std)
    df['desvio_padrao_weekday'].fillna(df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
    return df

def adiciona_media_desvio_por_janela_dia_semana_4(df):   
    df['time'] = pd.to_datetime(df['time'], format = '%Y-%m-%d %H:%M:%S')
    df['date'] = df['time'].dt.date
    mask4 = (df['time'] > '2016-09-18 00:00:00') & (df['time'] < '2016-10-25 00:00:00')
    df = df.loc[mask4]
    df['min_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.min)
    df['max_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.max)
    df['mediana_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.median)
    df['media_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean)
    df['desvio_padrao_weekday'] = df.groupby(['time_window', 'week','direction', 'tollgate_id'])["volume"].transform(np.std)
    df['desvio_padrao_weekday'].fillna(df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
    return df

def adiciona_media_desvio_por_janela_dia_semana_5_teste(df1, df2):
    df_list = [df1, df2]
    df = pd.concat(df_list, ignore_index=True)
    df['time'] = pd.to_datetime(df['time'], format = '%Y-%m-%d %H:%M:%S')
    df['date'] = df['time'].dt.date
    mask4 = (df['time'] >= '2016-10-18 00:00:00') & (df['time'] < '2016-11-01 00:00:00')
    df = df.loc[mask4]
    df['min_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.min)
    df['max_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.max)
    df['mediana_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.median)
    df['media_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean)
    df['desvio_padrao_weekday'] = df.groupby(['time_window', 'week','direction', 'tollgate_id'])["volume"].transform(np.std)
    df['desvio_padrao_weekday'].fillna(df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
    
    
    return df


# In[59]:


df_train4_a_25 = adiciona_media_desvio_por_janela_dia_semana_1(df_train4)
df_train4_a_10 = adiciona_media_desvio_por_janela_dia_semana_2(df_train4)
df_train4_a_17 = adiciona_media_desvio_por_janela_dia_semana_3(df_train4)
df_train4_a_24 = adiciona_media_desvio_por_janela_dia_semana_4(df_train4)
df_teste_a_24 = adiciona_media_desvio_por_janela_dia_semana_5_teste(df_teste_para_weekday, df_test1)


# In[60]:


len4_train1 = len(df_train4_a_25)
len4_train2 = len(df_train4_a_10)
len4_train3 = len(df_train4_a_17)
len4_train4 = len(df_train4_a_24)

x1 = df_train4_a_25.ix[:len4_train1 - 1, 23:]
x2 = df_train4_a_10.ix[:len4_train2 - 1, 23:]
x3 = df_train4_a_17.ix[:len4_train3 - 1, 23:]
x4 = df_train4_a_24.ix[:len4_train4 - 1, 23:]

df_train_list4 = [x1, x2, x3, x4]


# In[91]:


x4.head()


# In[61]:


df_train_para_agregar4 = pd.concat(df_train_list4, ignore_index=True)


# In[62]:


df_train4['media_volume_weekday'] = df_train_para_agregar4['media_volume_weekday']
df_train4['min_volume_weekday'] = df_train_para_agregar4['min_volume_weekday']
df_train4['max_volume_weekday'] = df_train_para_agregar4['max_volume_weekday']
df_train4['desvio_padrao_weekday'] = df_train_para_agregar4['desvio_padrao_weekday']
df_train4['mediana_volume_weekday'] = df_train_para_agregar4['mediana_volume_weekday']


# In[94]:


df_train4.to_csv('result_final/train4_final.csv', index=False)


# In[71]:


df_test1['min_volume_weekday'] = df_teste_a_24['min_volume_weekday']
df_test1['max_volume_weekday'] = df_teste_a_24['max_volume_weekday']
df_test1['mediana_volume_weekday'] = df_teste_a_24['mediana_volume_weekday']
df_test1['media_volume_weekday'] = df_teste_a_24['media_volume_weekday']
df_test1['desvio_padrao_weekday'] = df_teste_a_24['desvio_padrao_weekday']


# In[72]:


df_test1.to_csv('result_final/test_final.csv', index=False)


# In[14]:


df_train_list = [df_train1,df_train2, df_train3, df_train4]
#df_test_list = [df_test1, df_test2, df_test3, df_test4]


# In[15]:


random.shuffle(df_train_list)
df_train = pd.concat(df_train_list)


# In[164]:


df_train_grouped = df_train.groupby(["tollgate_id", "direction"])
df_test_grouped = df_test.groupby(["tollgate_id", "direction"])
result = []
oob = []
for key, train_data in df_train_grouped:
    test_data = df_test_grouped.get_group(key)
    len_train = len(train_data)
    regressor_cubic = RandomForestRegressor(n_estimators=500, max_depth=6)
    train_data = pd.DataFrame.reset_index(train_data)
    train_data = train_data.drop("index", axis=1)
    y = train_data.ix[:len_train - 1, :]["volume"]

    x = train_data[['week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday']]
    x1 = test_data[['week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday']]
    regressor_cubic.fit(x, y)
    yhat = regressor_cubic.predict(x1)
    
    test_data["volume"] = yhat
    result.append(test_data[['tollgate_id', 'time', 'direction', 'volume']])


df_result = pd.concat(result, axis=0)
#df_result.to_csv("result_final/result_predict_agora_rf.csv", index=False)


# In[112]:


train_data.columns


# In[109]:


x1


# In[100]:


df_test = df_test1


# In[101]:


df_remove = df_test.loc[(df_test['volume'] == 0)]

df_test = df_test.drop(df_remove.index)


# In[152]:


df_result_predict = pd.read_csv('result_final/result_gy_agora_matheus.csv')


# In[165]:


values_reais = df_test['volume'].values
values_predict = df_result['volume'].values


# In[155]:


df_test.head()


# In[156]:


df_result_predict.head()


# In[2]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# In[166]:


mean_absolute_percentage_error(values_reais, values_predict)


# In[167]:


rmse = sqrt(mean_squared_error(values_reais, values_predict))
rmse


# In[151]:


path = "result_models"
file_list = Test2(path)
df = pd.read_csv(file_list[0])

file_list.remove(file_list[0])
for x in file_list:
    dftmp = pd.read_csv(x)
    df = df.merge(dftmp, on=["tollgate_id", "time", "direction"])

result_list = []
for index, row in df.iterrows():
    volume_list = row[3:].tolist()
    # print volume_list
    volume_list1 = sorted(volume_list)

    result = np.mean([volume_list1[0], volume_list1[1]])
    # result = np.mean(volume_list1)

    result_list.append(result)
df = df[["tollgate_id", "time", "direction"]]
df["volume"] = result_list

df["time_window_start"] = pd.to_datetime(df["time"])
df["time_window_end"] = df["time_window_start"] + timedelta(minutes=20)
list_tw = []
for x in range(0, len(df["time_window_start"] )):
    str_tw =  '[' + str(df["time_window_start"][x]) + ',' + str(df["time_window_end"][x]) + ')'
    list_tw.append(str_tw)

df["time"] = list_tw
df = df[["tollgate_id", "time", "direction","volume"]]



path  =  os.path.dirname( os.getcwd())
df.to_csv("result_final/result_gy_agora_matheus.csv", index=False)


# In[137]:


def Test2(rootDir):
    file_list = []
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        file_list.append(path)
        if os.path.isdir(path):
            Test2(path)
    return file_list


# In[3]:


df_train_final_1 = pd.read_csv('result_final/train1_final.csv')
df_train_final_2 = pd.read_csv('result_final/train2_final.csv')
df_train_final_3 = pd.read_csv('result_final/train3_final.csv')
df_train_final_4 = pd.read_csv('result_final/train4_final.csv')


# In[4]:


df_test_final = pd.read_csv('result_final/test_final.csv')
df_remove = df_test_final.loc[(df_test_final['volume'] == 0)]

df_test_final = df_test_final.drop(df_remove.index)


# In[5]:


df_train_list_final = [df_train_final_1,df_train_final_2, df_train_final_3, df_train_final_4]


# In[6]:


random.shuffle(df_train_list_final)
df_train = pd.concat(df_train_list_final)


# In[8]:


df_train.to_csv('verificar_ordem.csv')


# In[7]:


df_test_final.head()


# In[7]:


def feature_format():
    #pd_volume_train = pd_volume_train.set_index(['time'])
    #pd_volume_test = pd_volume_test.set_index(['time'])
    #volume_train = v_train.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #volume_test = v_test.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #print(volume_train)                
    x = pd.Series(df_train['time_window'].unique())
    s = pd.Series(range(len(x)),index = x.values)
    df_train['window_n'] = df_train['time_window'].map(s)
    df_test_final['window_n'] = df_test_final['time_window'].map(s)
#        print vol_test.tail()
    #volume_train['weekday'] = v_train['weekday']
    #volume_test['weekday'] = v_test['weekday']
    
    feature_train = df_train.drop('volume', axis = 1)
    feature_test = df_test_final.drop('volume',axis = 1)
    values_train = df_train['volume'].values
    values_test = df_test_final['volume'].values
    
    return feature_train, feature_test, values_train, values_test


# In[8]:


feature_train, feature_test, values_train, values_test = feature_format()
feature_train = pd.concat([feature_train, pd.get_dummies(feature_train['tollgate_id'])], axis=1)
feature_test = pd.concat([feature_test, pd.get_dummies(feature_test['tollgate_id'])], axis=1)
#feature_train = pd.concat([feature_train, pd.get_dummies(feature_train['week'])], axis=1)
#feature_test = pd.concat([feature_test, pd.get_dummies(feature_test['week'])], axis=1)


# In[26]:


feature_train.to_csv('verificar_ordem_dados.csv', index=False)


# In[28]:


feature_train.head()


# In[9]:


regressor = RandomForestRegressor(n_estimators = 800, max_depth=20, max_features=None)


# In[10]:


regressor.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday']], values_train)


# In[74]:


y_pred = regressor.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday']])


# In[16]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) 


# In[76]:


mean_absolute_percentage_error(values_test, y_pred)


# In[77]:


sqrt(mean_squared_error(values_test, y_pred))


# In[79]:


regressor.score(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday']], values_train)


# In[80]:


regressor.score(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday']], values_test)


# In[81]:


regressor.feature_importances_


# In[ ]:


scores_ada_2 = cross_val_score(regressor,feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday']],values_train, scoring="neg_mean_squared_error",cv=4, n_jobs=-1)
tree_rmse_scores_test_ada_2 = np.sqrt(-scores_ada_2)


# In[ ]:


display_scores(tree_rmse_scores_test_ada)


# In[36]:


def display_scores(scores):
    print("Scores:", (scores))
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[29]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [1,2,3,4,5,6,7,8,9,10,15,20],
    'max_features': [None],
    'n_estimators': [200, 300, 500, 800, 1200, 1500]
    #colocar um intervalo mais logico. Treino teste e validacao (tira uma parte do treino para validacao, e os
    # e eu nao posso fazer isso pra proxima janela de treino)
    #volume proximo nao pode ser utilizado
    # a media de dias da semana anterior
    # max altura, max numero de features, n_estimadores (100, 200, 300, 400)
    # max altura (80, 90,100, 110)
    # n_features deixa o numero de features fixo que eu tenho passa o auto
    # random_state tirar do decision tree
    #fazer isso
    #falar primeiro do decision tree antes mesmo do random forest
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 4, n_jobs = -1, verbose = 2)


# In[30]:


("")


# In[31]:


grid_search.best_params_


# In[12]:


regressor_rf = RandomForestRegressor(n_estimators = 800, max_depth=20, max_features=None)


# In[13]:


regressor_rf.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[55]:


regressor_rf.score(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[56]:


regressor_rf.score(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_test)


# In[15]:


y_pred_rf = regressor_rf.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])


# In[16]:


mean_absolute_percentage_error(values_test, y_pred_rf)


# In[17]:


regressor_rf.score(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_test)


# In[18]:


sqrt(mean_squared_error(values_test, y_pred_rf))


# In[21]:


mean_absolute_error(y_pred_rf, values_test)


# In[19]:


regressor_rf.feature_importances_


# In[32]:


feat_importances = pd.Series(regressor_rf.feature_importances_, index=feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']].columns)
feat_importances.nlargest(25).plot(kind='barh')
plt.rcParams['figure.figsize'] = (11,8)
plt.xscale('log')


# In[192]:


plt.scatter(y_pred_rf, values_test)


# In[69]:


denominator_rf = y_pred_rf.dot(y_pred_rf) - y_pred_rf.mean() * y_pred_rf.sum()


# In[70]:


m_rf = ( y_pred_rf.dot(values_test) - values_test.mean() * y_pred_rf.sum()) / denominator_rf

b_rf = ( values_test.mean() * y_pred_rf.dot(y_pred_rf) - y_pred_rf.mean() * y_pred_rf.dot(values_test)) / denominator_rf


# In[71]:


pred_rf = m_rf * y_pred_rf + b_rf


# In[72]:


plt.scatter(y_pred_rf, values_test)
plt.plot(y_pred_rf, pred_rf, 'r')


# In[73]:


res_rf = values_test - y_pred_rf
tot_rf = values_test - values_test.mean()
R_squared_rf = 1 - res_rf.dot(res_rf) / tot_rf.dot(tot_rf)
print(R_squared_rf)


# In[51]:


scores_rf = cross_val_score(regressor_rf,feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']],values_train, scoring="neg_mean_squared_error",cv=4)
tree_rmse_scores_test_rf = np.sqrt(-scores_rf)


# In[54]:


#Com CV = 4
display_scores(tree_rmse_scores_test_rf)


# In[ ]:


#Com CV = 10
display_scores(tree_rmse_scores_test_rf)


# In[37]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [1,2,3,4,5,6,7,8,9,10,15,20],
    'max_features': [None],
    'n_estimators': [200, 300, 500, 800, 1200, 1500]
    #colocar um intervalo mais logico. Treino teste e validacao (tira uma parte do treino para validacao, e os
    # e eu nao posso fazer isso pra proxima janela de treino)
    #volume proximo nao pode ser utilizado
    # a media de dias da semana anterior
    # max altura, max numero de features, n_estimadores (100, 200, 300, 400)
    # max altura (80, 90,100, 110)
    # n_features deixa o numero de features fixo que eu tenho passa o auto
    # random_state tirar do decision tree
    #fazer isso
    #falar primeiro do decision tree antes mesmo do random forest
}
# Create a based model
gbrt = GradientBoostingRegressor()
# Instantiate the grid search model
grid_search_gbrt = GridSearchCV(estimator = gbrt, param_grid = param_grid, 
                          cv = 4, n_jobs = -1, verbose = 2)


# In[38]:


grid_search_gbrt.fit(feature_trainfeature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[39]:


grid_search_gbrt.best_params_


# In[22]:


regressor_gbrt = GradientBoostingRegressor(n_estimators=1500,max_depth=9)


# In[38]:


regressor_gbrt.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[39]:


y_pred_gbrt = regressor_gbrt.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])


# In[40]:


mean_absolute_percentage_error(values_test, y_pred_gbrt)


# In[48]:


sqrt(mean_squared_error(values_test, y_pred_gbrt))


# In[42]:


mean_absolute_error(values_test, y_pred_gbrt)


# In[43]:


regressor_gbrt.score(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[44]:


regressor_gbrt.score(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_test)


# In[43]:


regressor_gbrt.feature_importances_


# In[44]:


feat_importances_gbrt = pd.Series(regressor_gbrt.feature_importances_, index=feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']].columns)
feat_importances_gbrt.nlargest(25).plot(kind='barh')
plt.rcParams['figure.figsize'] = (11,8)
plt.xscale('log')


# In[45]:


scores_gbrt = cross_val_score(regressor_gbrt,feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']],values_train, scoring="neg_mean_squared_error",cv=4)
tree_rmse_scores_test_gbrt = np.sqrt(-scores_gbrt)


# In[46]:


def display_scores(scores):
    print("Scores:", (scores))
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[49]:


display_scores(tree_rmse_scores_test_gbrt)


# In[67]:


denominator = y_pred_gbrt.dot(y_pred_gbrt) - y_pred_gbrt.mean() * y_pred_gbrt.sum()

m_gbrt = ( y_pred_gbrt.dot(values_test) - values_test.mean() * y_pred_gbrt.sum()) / denominator

b_gbrt = ( values_test.mean() * y_pred_gbrt.dot(y_pred_gbrt) - y_pred_gbrt.mean() * y_pred_gbrt.dot(values_test)) / denominator

pred_gbrt = m_gbrt * y_pred_gbrt + b_gbrt

plt.scatter(y_pred_gbrt, values_test)
plt.plot(y_pred_gbrt, pred_gbrt, 'r')


# In[75]:


res_gbrt = values_test - y_pred_gbrt
tot_gbrt = values_test - values_test.mean()
R_squared_gbrt = 1 - res_gbrt.dot(res_gbrt) / tot_gbrt.dot(tot_gbrt)
print(R_squared_gbrt)


# In[44]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [1,2,3,4,5,6,7,8,9,10,15,20],
    #colocar um intervalo mais logico. Treino teste e validacao (tira uma parte do treino para validacao, e os
    # e eu nao posso fazer isso pra proxima janela de treino)
    #volume proximo nao pode ser utilizado
    # a media de dias da semana anterior
    # max altura, max numero de features, n_estimadores (100, 200, 300, 400)
    # max altura (80, 90,100, 110)
    # n_features deixa o numero de features fixo que eu tenho passa o auto
    # random_state tirar do decision tree
    #fazer isso
    #falar primeiro do decision tree antes mesmo do random forest
}
# Create a based model
dec = DecisionTreeRegressor()

# Instantiate the grid search model
grid_search_dec = GridSearchCV(estimator = dec, param_grid = param_grid, 
                          cv = 4, n_jobs = -1, verbose = 2)


# In[45]:


grid_search_dec.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday']], values_train)


# In[46]:


grid_search_dec.best_params_


# In[63]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'base_estimator': [DecisionTreeRegressor(max_depth=6)],
    'n_estimators': [200, 300, 500, 800, 1200, 1500]
    #colocar um intervalo mais logico. Treino teste e validacao (tira uma parte do treino para validacao, e os
    # e eu nao posso fazer isso pra proxima janela de treino)
    #volume proximo nao pode ser utilizado
    # a media de dias da semana anterior
    # max altura, max numero de features, n_estimadores (100, 200, 300, 400)
    # max altura (80, 90,100, 110)
    # n_features deixa o numero de features fixo que eu tenho passa o auto
    # random_state tirar do decision tree
    #fazer isso
    #falar primeiro do decision tree antes mesmo do random forest
}
# Create a based model
ada = AdaBoostRegressor()

# Instantiate the grid search model
grid_search_ada = GridSearchCV(estimator = ada, param_grid = param_grid, 
                          cv = 4, n_jobs = -1, verbose = 2)


# In[64]:


grid_search_ada.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday']], values_train)


# In[65]:


grid_search_ada.best_params_


# In[26]:


regressor_ada = AdaBoostRegressor(DecisionTreeRegressor(criterion='mse', max_depth=6, max_features=None,
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=0, splitter='best'), n_estimators = 300)


# In[27]:


regressor_ada.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[28]:


y_pred_ada = regressor_ada.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])


# In[29]:


mean_absolute_percentage_error(values_test, y_pred_ada)


# In[30]:


sqrt(mean_squared_error(values_test, y_pred_ada))


# In[31]:


mean_absolute_error(values_test, y_pred_ada)


# In[32]:


regressor_ada.score(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[33]:


regressor_ada.score(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_test)


# In[50]:


scores_ada = cross_val_score(regressor_ada,feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']],values_train, scoring="neg_mean_squared_error",cv=4)
tree_rmse_scores_test_ada = np.sqrt(-scores_ada)


# In[37]:


display_scores(tree_rmse_scores_test_ada)


# In[59]:


#Com o CV = 10
display_scores(tree_rmse_scores_test_ada)


# In[60]:


feat_importances_ada = pd.Series(regressor_ada.feature_importances_, index=feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']].columns)
feat_importances_ada.nlargest(25).plot(kind='barh')
plt.rcParams['figure.figsize'] = (11,8)
plt.xscale('log')


# In[68]:


denominator_ada = y_pred_ada.dot(y_pred_ada) - y_pred_ada.mean() * y_pred_ada.sum()

m_ada = ( y_pred_ada.dot(values_test) - values_test.mean() * y_pred_ada.sum()) / denominator_ada

b_ada = ( values_test.mean() * y_pred_ada.dot(y_pred_ada) - y_pred_ada.mean() * y_pred_ada.dot(values_test)) / denominator_ada

pred_ada = m_ada * y_pred_ada + b_ada

plt.scatter(y_pred_ada, values_test)
plt.plot(y_pred_ada, pred_ada, 'r')


# In[74]:


res_ada = values_test - y_pred_ada
tot_ada = values_test - values_test.mean()
R_squared_ada = 1 - res_ada.dot(res_ada) / tot_ada.dot(tot_ada)
print(R_squared_ada)

