
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from datetime import time
import matplotlib.pyplot as pplot
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
from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.legend_handler import HandlerLine2D
from sklearn.learning_curve import learning_curve


# In[20]:


df_train = pd.read_csv('data_process_final/treino_final.csv')
df_teste = pd.read_csv('data_process_final/teste_final.csv')
df_teste_para_weekday = pd.read_csv('data_process_final/teste_final_para_weekday.csv')


# In[200]:


df_validation_1 = df_remove[((df_remove['hour'] >= 6) & (df_remove['hour'] <= 9))]


# In[11]:


df_validation_2 = df_remove[((df_remove['hour'] >= 15) & (df_remove['hour'] <= 16))]


# In[206]:


volumes_train_validation = df_validation_1['volume'].values


# In[3]:


df_validation_list = [df_validation_1, df_validation_2]
df_validation = pd.concat(df_validation_list, ignore_index=True)


# In[14]:


df_validation.to_csv('data_validation.csv', index=False)


# In[21]:


df_remove = df_train.loc[(df_train['day'] >= 1) & (df_train['day'] <= 7) ]

df_train = df_train.drop(df_remove.index)


# In[22]:


del df_train['volume_proximo']
del df_teste['volume_proximo']


# In[23]:


del df_train['volume_proximo_2']
del df_teste['volume_proximo_2']


# In[24]:


def adiciona_media_desvio_por_dia(df):
    df['media_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])["volume"].transform(np.mean)
    df['desvio_padrao_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])["volume"].transform(np.std)
    df['desvio_padrao_hora_dia'].fillna(df.groupby(['direction', 'tollgate_id', 'day', 'hour'])["volume"].transform(np.mean), inplace=True)
    df['min_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])['volume'].transform(np.min)
    df['max_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])['volume'].transform(np.max)
    df['mediana_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])['volume'].transform(np.median)
    return df


# In[25]:


df_train = adiciona_media_desvio_por_dia(df_train)
df_teste= adiciona_media_desvio_por_dia(df_teste)


# In[26]:


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


# In[27]:


df_train_a_25 =adiciona_media_desvio_por_janela_dia_semana_1(df_train)
df_train_a_10 = adiciona_media_desvio_por_janela_dia_semana_2(df_train)
df_train_a_17 = adiciona_media_desvio_por_janela_dia_semana_3(df_train)
df_train_a_24 = adiciona_media_desvio_por_janela_dia_semana_4(df_train)
df_teste_a_24 = adiciona_media_desvio_por_janela_dia_semana_5_teste(df_teste_para_weekday, df_teste)


# In[28]:


len_train1 = len(df_train_a_25)
len_train2 = len(df_train_a_10)
len_train3 = len(df_train_a_17)
len_train4 = len(df_train_a_24)

x1 = df_train_a_25.ix[:len_train1 - 1, 21:]
x2 = df_train_a_10.ix[:len_train2 - 1, 21:]
x3 = df_train_a_17.ix[:len_train3 - 1, 21:]
x4 = df_train_a_24.ix[:len_train4 - 1, 21:]

df_train_list = [x1, x2, x3, x4]

#df_train_list = [df_train_a_25[df_train_a_25[['min_volume_weekday']]], df_train_a_10[df_train_a_10['min_volume_weekday']]]
#df_train_para_agregar = pd.concat(df_train_list)


# In[29]:


df_train_para_agregar = pd.concat(df_train_list, ignore_index=True)


# In[30]:


df_train['media_volume_weekday'] = df_train_para_agregar['media_volume_weekday']
df_train['min_volume_weekday'] = df_train_para_agregar['min_volume_weekday']
df_train['max_volume_weekday'] = df_train_para_agregar['max_volume_weekday']
df_train['desvio_padrao_weekday'] = df_train_para_agregar['desvio_padrao_weekday']
df_train['mediana_volume_weekday'] = df_train_para_agregar['mediana_volume_weekday']


# In[31]:


df_teste['min_volume_weekday'] = df_teste_a_24['min_volume_weekday']
df_teste['max_volume_weekday'] = df_teste_a_24['max_volume_weekday']
df_teste['mediana_volume_weekday'] = df_teste_a_24['mediana_volume_weekday']
df_teste['media_volume_weekday'] = df_teste_a_24['media_volume_weekday']
df_teste['desvio_padrao_weekday'] = df_teste_a_24['desvio_padrao_weekday']


# In[ ]:


#media do volume do dia naquela tollgate naquela direcao


# In[32]:


def medidas_volume_tollgate_direction_am_pm(df):
    df['min_volume_dia_am_pm'] = df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.min)
    df['max_volume_dia_am_pm'] = df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.max)
    df['mediana_volume_dia_am_pm'] = df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.median)
    df['media_volume_dia_am_pm'] = df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.mean)
    df['desvio_padrao_am_pm'] = df.groupby(['day','direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.std)
    df['desvio_padrao_am_pm'].fillna(df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.mean), inplace=True)
    return df


# In[33]:


df_train = medidas_volume_tollgate_direction_am_pm(df_train)
df_teste = medidas_volume_tollgate_direction_am_pm(df_teste)


# In[70]:


df_train_am = df_train[df_train['am_pm'] == 0]
df_teste_am = df_teste[df_teste['am_pm'] == 0]


# In[76]:


def feature_format_am():
    #pd_volume_train = pd_volume_train.set_index(['time'])
    #pd_volume_test = pd_volume_test.set_index(['time'])
    #volume_train = v_train.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #volume_test = v_test.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #print(volume_train)                
    x = pd.Series(df_train_am['time_window'].unique())
    s = pd.Series(range(len(x)),index = x.values)
    df_train_am['window_n'] = df_train_am['time_window'].map(s)
    df_teste_am['window_n'] = df_teste_am['time_window'].map(s)
#        print vol_test.tail()
    #volume_train['weekday'] = v_train['weekday']
    #volume_test['weekday'] = v_test['weekday']
    
    feature_train_am = df_train_am.drop('volume', axis = 1)
    feature_test_am = df_teste_am.drop('volume',axis = 1)
    values_train_am = df_train_am['volume'].values
    values_test_am = df_teste_am['volume'].values
    
    return feature_train_am, feature_test_am, values_train_am, values_test_am


# In[77]:


feature_train_am, feature_test_am, values_train_am, values_test_am = feature_format_am()


# In[141]:


regressor_cubic_am = RandomForestRegressor(n_estimators=500, max_depth=4, random_state=10, oob_score=True)


# In[142]:


regressor_cubic_am.fit(feature_train_am[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday', 'window_n']], values_train_am)


# In[143]:


y_pred_am = regressor_cubic_am.predict(feature_test_am[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday', 'window_n']])


# In[144]:


mean_absolute_percentage_error(values_test_am, y_pred_am)


# In[145]:


rmse = sqrt(mean_squared_error(y_pred_am, values_test_am))
rmse


# In[ ]:


#regr_ada.score(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday', 'window_n']], values_train)


# In[146]:


regressor_cubic_am.score(feature_train_am[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday', 'window_n']],values_train_am)


# In[147]:


regressor_cubic_am.score(feature_test_am[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday', 'window_n']],values_test_am)


# In[132]:


regressor_cubic_am.feature_importances_


# In[226]:


df_train.to_csv('train_final.csv', index=False)
df_teste.to_csv('teste_final.csv', index=False)


# In[34]:


def feature_format():
    #pd_volume_train = pd_volume_train.set_index(['time'])
    #pd_volume_test = pd_volume_test.set_index(['time'])
    #volume_train = v_train.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #volume_test = v_test.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #print(volume_train)                
    x = pd.Series(df_train['time_window'].unique())
    s = pd.Series(range(len(x)),index = x.values)
    df_train['window_n'] = df_train['time_window'].map(s)
    df_teste['window_n'] = df_teste['time_window'].map(s)
#        print vol_test.tail()
    #volume_train['weekday'] = v_train['weekday']
    #volume_test['weekday'] = v_test['weekday']
    
    feature_train = df_train.drop('volume', axis = 1)
    feature_test = df_teste.drop('volume',axis = 1)
    values_train = df_train['volume'].values
    values_test = df_teste['volume'].values
    
    return feature_train, feature_test, values_train, values_test


# In[35]:


feature_train, feature_test, values_train, values_test = feature_format()
feature_train = pd.concat([feature_train, pd.get_dummies(feature_train['tollgate_id'])], axis=1)
feature_test = pd.concat([feature_test, pd.get_dummies(feature_test['tollgate_id'])], axis=1)


# In[417]:


df_train.to_csv('train_final_oficial.csv', index=False)
df_teste.to_csv('teste_final_oficial.csv', index=False)


# In[43]:


regressor_cubic = RandomForestRegressor(n_estimators=1200, max_depth=10, random_state=10, max_features='sqrt', oob_score=True,warm_start=True)


# In[44]:


regressor_cubic.fit(feature_train[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'window_n']], values_train)


# In[45]:


y_pred = regressor_cubic.predict(feature_test[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'window_n']])


# In[46]:


mean_absolute_percentage_error(values_test, y_pred)


# In[47]:


rmse = sqrt(mean_squared_error(y_pred, values_test))
rmse


# In[36]:


scores = cross_val_score(regressor_cubic,feature_train[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'window_n']],values_train, scoring="neg_mean_squared_error",cv=5)
tree_rmse_scores = np.sqrt(-scores)


#[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday']]


# In[56]:


def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard	deviation:", scores.std())


# In[38]:


display_scores(tree_rmse_scores)


# In[39]:


scores_test = cross_val_score(regressor_cubic,feature_test[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'window_n']],values_test, scoring="neg_mean_squared_error",cv=5)
tree_rmse_scores_test = np.sqrt(-scores_test)


# In[67]:


display_scores(tree_rmse_scores_test)


# In[48]:


regressor_cubic.score(feature_train[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'window_n']],values_train)


# In[49]:


regressor_cubic.score(feature_test[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'window_n']],values_test)


# In[50]:


regressor_cubic.feature_importances_


# In[125]:


regressor_cubic_2.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[121]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [2,4,6,8],
    'min_samples_leaf': [1,2,4,6],
    'min_samples_split': [2,4,6,8],
    'n_estimators': [600, 900, 1200, 1500],
    'oob_score': [True]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 4, n_jobs = -1, verbose = 2)


# In[122]:


grid_search.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[123]:


grid_search.best_params_


# In[382]:


regressor_cubic_2 = RandomForestRegressor(n_estimators=1500, max_depth=8, max_features=None, min_samples_leaf=1, min_samples_split=4, oob_score=True)


# In[383]:


regressor_cubic_2.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[384]:


y_pred_2 = regressor_cubic_2.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])


# In[385]:


mean_absolute_error(values_test, y_pred_2)


# In[386]:


rmse = sqrt(mean_squared_error(y_pred_2, values_test))
rmse


# In[387]:


y_train_predicted_2 = regressor_cubic_2.predict(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
y_test_predicted_pruned_trees_2 = regressor_cubic_2.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
rmse_train_2 = sqrt(mean_squared_error(values_train, y_train_predicted_2))
rmse_test_2 = sqrt(mean_squared_error(values_test, y_test_predicted_pruned_trees_2))
print("RF with pruned trees, Train MSE: {} Test MSE: {}".format(rmse_train_2, rmse_test_2))


# In[388]:


mean_absolute_percentage_error(values_test, y_pred_2)


# In[389]:


regressor_cubic_2.score(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[390]:


regressor_cubic_2.score(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_test)


# In[391]:


regressor_cubic_2.feature_importances_


# In[392]:


scores_2 = cross_val_score(regressor_cubic_2,feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']],values_train, scoring="neg_mean_squared_error",cv=4)
tree_rmse_scores_test_2 = np.sqrt(-scores_2)


# In[393]:


display_scores(tree_rmse_scores_test_2)


# In[394]:


y_train_predicted_rf_2_mape = regressor_cubic_2.predict(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
y_test_predicted_pruned_trees_rf_2_mape = regressor_cubic_2.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
mape_train_rf_2 = mean_absolute_percentage_error(values_train, y_train_predicted_rf_2_mape)
mape_test_rf_2 = mean_absolute_percentage_error(values_test, y_test_predicted_pruned_trees_rf_2_mape)
print("RF_2 with pruned trees, Train MAPE: {} Test MAPE: {}".format(mape_train_rf_2, mape_test_rf_2))


# In[395]:


feat_importances_rf_2 = pd.Series(regressor_cubic_2.feature_importances_, index=feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']].columns)
feat_importances_rf_2.nlargest(25).plot(kind='barh')
plt.rcParams['figure.figsize'] = (11,8)
plt.xscale('log')


# In[355]:


# Create the parameter grid based on the results of random search 
param_grid_gbrt = {
    'max_depth': [3,4,6,8],
    'min_samples_leaf': [1,2,4,6],
    'min_samples_split': [2,4,6,8],
    'n_estimators': [100, 200, 300, 400, 600],
    'learning_rate': [1, 0.5, 0.25, 0.1, 0.05, 0.01]
}
# Create a based model
gbrt = GradientBoostingRegressor()
# Instantiate the grid search model
grid_search_gbrt = GridSearchCV(estimator = gbrt, param_grid = param_grid_gbrt, 
                          cv = 4, n_jobs = -1, verbose = 2)


# In[356]:


grid_search_gbrt.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[357]:


grid_search_gbrt.best_params_


# In[ ]:


#{'max_depth': 10, 'max_features': None, 'n_estimators': 1200}


# In[36]:





# In[21]:


regressor_cubic = RandomForestRegressor(max_depth= 10,n_estimators= 1200, oob_score=True)


# In[363]:


regressor_cubic.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[364]:


y_pred = regressor_cubic.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])


# In[365]:


mean_absolute_percentage_error(values_test, y_pred)


# In[366]:


rmse = sqrt(mean_squared_error(y_pred, values_test))
rmse


# In[367]:


mean_absolute_error(values_test, y_pred)


# In[29]:


regressor_cubic.feature_importances_


# In[374]:


media_valores = np.std(y_pred)


# In[375]:


media_valores


# In[30]:


regressor_cubic.score(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[31]:


regressor_cubic.score(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_test)


# In[32]:


regressor_cubic.feature_importances_


# In[33]:


denominator = y_pred.dot(y_pred) - y_pred.mean() * y_pred.sum()


# In[34]:


m = ( y_pred.dot(values_test) - values_test.mean() * y_pred.sum()) / denominator

b = ( values_test.mean() * y_pred.dot(y_pred) - y_pred.mean() * y_pred.dot(values_test)) / denominator


# In[35]:


pred = m * y_pred + b


# In[416]:


plt.scatter(y_pred, values_test)
plt.xlabel("Volume previsto pelo modelo") 
plt.ylabel("Volume observado")
plt.plot(y_pred, pred, 'r')


# In[37]:


res = values_test - y_pred
tot = values_test - values_test.mean()


# In[38]:


R_squared = 1 - res.dot(res) / tot.dot(tot)


# In[39]:


print(R_squared)


# In[40]:


def display_scores(scores):
    print("Scores:", (scores))
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# In[121]:


scores = cross_val_score(regressor_cubic,feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']],values_train, scoring="neg_mean_squared_error",cv=4)
tree_rmse_scores_test = np.sqrt(-scores)


# In[122]:


display_scores(tree_rmse_scores_test)


# In[ ]:


df_p = performance_metrics(df_cv)
print(df_p)
from fbprophet.plot import plot_cross_validation_metric
plot_cross_validation_metric(df_cv, metric='mape').savefig('test_mape.png')


# In[361]:


#colocar o grafico em escala logaritimica
feat_importances = pd.Series(regressor_cubic.feature_importances_, index=feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']].columns)
feat_importances.nlargest(25).plot(kind='barh')
plt.rcParams['figure.figsize'] = (11,8)
plt.xscale('log')


# In[369]:


y_train_predicted = regressor_cubic.predict(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
y_test_predicted_pruned_trees = regressor_cubic.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
mse_train = sqrt(mean_squared_error(values_train, y_train_predicted))
mse_test = sqrt(mean_squared_error(values_test, y_test_predicted_pruned_trees))
print("RF with pruned trees, Train RMSE: {} Test RMSE: {}".format(mse_train, mse_test))


# In[368]:


y_train_predicted_rf_mape = regressor_cubic.predict(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
y_test_predicted_pruned_trees_rf_mape = regressor_cubic.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
mape_train_rf = mean_absolute_percentage_error(values_train, y_train_predicted_rf_mape)
mape_test_rf = mean_absolute_percentage_error(values_test, y_test_predicted_pruned_trees_rf_mape)
print("RF with pruned trees, Train MAPE: {} Test MAPE: {}".format(mape_train_rf, mape_test_rf))


# In[398]:


title = "Learning Curves (Random Forest)" 
estimator = RandomForestRegressor(n_estimators=regressor_cubic.n_estimators, max_depth=regressor_cubic.max_depth) 
plot_learning_curve(estimator, title, feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train, cv=4, n_jobs=-1) 
plt.show() 


# In[397]:


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)): 
    plt.figure() 
    plt.title(title) 
    if ylim is not None: 
        plt.ylim(*ylim) 
    plt.xlabel("Training examples") 
    plt.ylabel("Score") 
    train_sizes, train_scores, test_scores = learning_curve( estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes) 
    train_scores_mean = np.mean(train_scores, axis=1) 
    train_scores_std = np.std(train_scores, axis=1) 
    test_scores_mean = np.mean(test_scores, axis=1) 
    test_scores_std = np.std(test_scores, axis=1) 
    plt.grid() 
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r") 
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g") 
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score") 
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score") 
    plt.legend(loc="best") 
    return plt 


# In[ ]:


#Adicionar um paragrafo falando sobre os resultados

#Pegar as 5 melhores features de cada modelo e colocar numa tabela


# In[ ]:


##ADABOOSTING REGRESSOR


# In[435]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth':[1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100],
    'splitter': ['best', 'random'],
}
# Create a based model
dec = DecisionTreeRegressor()
# Instantiate the grid search model
grid_search_dt = GridSearchCV(estimator = dec, param_grid = param_grid, 
                          cv = 4, n_jobs = -1, verbose = 2)


# In[436]:


grid_search_dt.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday']], values_train)


# In[437]:


grid_search_dt.best_params_


# In[438]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'base_estimator':[DecisionTreeRegressor(max_depth = 6)],
    'loss':['linear', 'square', 'exponential'],
    'n_estimators': [200, 300, 500, 800, 1200, 1500, 1800]
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


# In[440]:


grid_search_ada.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday']], values_train)


# In[441]:


grid_search_ada.best_params_


# In[44]:


regr_ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6 ),n_estimators=200, loss='square')


# In[45]:


regr_ada.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train) 


# In[46]:


y_pred_ada = regr_ada.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])


# In[47]:


mean_absolute_percentage_error(values_test, y_pred_ada)


# In[49]:


mean_absolute_error(values_test, y_pred_ada)


# In[50]:


rmse = sqrt(mean_squared_error(values_test, y_pred_ada))
rmse


# In[91]:


regr_ada.score(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[135]:


regr_ada.score(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_test)


# In[221]:


regr_ada.feature_importances_


# In[138]:


denominator_ada = y_pred_ada.dot(y_pred_ada) - y_pred_ada.mean() * y_pred_ada.sum()


# In[139]:


m = ( y_pred_ada.dot(values_test) - values_test.mean() * y_pred_ada.sum()) / denominator_ada

b = ( values_test.mean() * y_pred_ada.dot(y_pred_ada) - y_pred_ada.mean() * y_pred_ada.dot(values_test)) / denominator_ada


# In[140]:


pred_ada = m * y_pred_ada + b


# In[141]:


plt.scatter(y_pred_ada, values_test)
plt.plot(y_pred_ada, pred_ada, 'r')


# In[142]:


res_ada = values_test - y_pred_ada
tot_ada = values_test - values_test.mean()


# In[143]:


R_squared_ada = 1 - res_ada.dot(res_ada) / tot_ada.dot(tot_ada)


# In[144]:


print(R_squared_ada)


# In[145]:


scores_ada = cross_val_score(regr_ada,feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']],values_train, scoring="neg_mean_squared_error",cv=4)
tree_rmse_scores_test_ada = np.sqrt(-scores_ada)


# In[146]:


display_scores(tree_rmse_scores_test)


# In[377]:


feat_importances_ada = pd.Series(regr_ada.feature_importances_, index=feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']].columns)
feat_importances_ada.nlargest(25).plot(kind='barh')
plt.rcParams['figure.figsize'] = (11,8)
plt.xscale('log')


# In[92]:


y_train_predicted_adaboost = regr_ada.predict(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
y_test_predicted_pruned_trees_adaboost = regr_ada.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
rmse_train_ada = sqrt(mean_squared_error(values_train, y_train_predicted_adaboost))
rmse_test_ada = sqrt(mean_squared_error(values_test, y_test_predicted_pruned_trees_adaboost))
print("ADABOOST, Train RMSE: {} Test RMSE: {}".format(rmse_train_ada, rmse_test_ada))


# In[399]:


y_train_predicted_adaboost_mape = regr_ada.predict(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
y_test_predicted_pruned_trees_adaboost_mape = regr_ada.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
mape_train_adaboost = mean_absolute_percentage_error(values_train, y_train_predicted_adaboost_mape)
mape_test_adaboost = mean_absolute_percentage_error(values_test, y_test_predicted_pruned_trees_adaboost_mape)
print("ADABOOST, Train MAPE: {} Test MAPE: {}".format(mape_train_adaboost, mape_test_adaboost))


# In[401]:


title = "Learning Curves (ADABOOST)" 
estimator_ada = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 6 ),n_estimators=200, loss='square') 
plot_learning_curve(estimator_ada, title, feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train, cv=4, n_jobs=-1) 
plt.show() 


# In[ ]:


param_grid={'base_estimator': [DecisionTreeRegressor([1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100])],
            'n_estimators':[200, 300, 500, 800, 1200, 1500, 1800], 
            'learning_rate': [0.1, 0.05, 0.01, 0.005], 
            'loss':['linear', 'square', 'exponential']}
n_jobs=-1 

cv,best_est=ADABooster(param_grid, n_jobs)


# In[48]:


regr_ada_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 50 ),n_estimators=1200, learning_rate=0.01, loss='square')


# In[69]:


regr_ada_2.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday']], values_train) 


# In[50]:


regr_ada_2.feature_importances_


# In[72]:


y_pred_ada_2 = regr_ada_2.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday']])


# In[73]:


mean_absolute_percentage_error(values_test, y_pred_ada_2)


# In[74]:


rmse = sqrt(mean_squared_error(values_test, y_pred_ada_2))
rmse


# In[75]:


regr_ada_2.score(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday', 'window_n']], values_train)


# In[76]:


regr_ada_2.score(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm','media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday', 'window_n']], values_test)


# In[77]:


denominator_ada_2 = y_pred_ada_2.dot(y_pred_ada_2) - y_pred_ada_2.mean() * y_pred_ada_2.sum()


# In[78]:


m_ada_2 = ( y_pred_ada_2.dot(values_test) - values_test.mean() * y_pred_ada_2.sum()) / denominator_ada_2

b_ada_2 = ( values_test.mean() * y_pred_ada_2.dot(y_pred_ada_2) - y_pred_ada_2.mean() * y_pred_ada_2.dot(values_test)) / denominator_ada_2


# In[79]:


pred_ada_2 = m_ada_2 * y_pred_ada_2 + b_ada_2


# In[80]:


plt.scatter(y_pred_ada_2, values_test)
plt.plot(y_pred_ada_2, pred_ada_2, 'r')


# In[81]:


res_ada_2 = values_test - y_pred_ada_2
tot_ada_2 = values_test - values_test.mean()


# In[82]:


R_squared_ada_2 = 1 - res_ada_2.dot(res_ada_2) / tot_ada_2.dot(tot_ada_2)


# In[83]:


print(R_squared_ada_2)


# In[84]:


scores_ada_2 = cross_val_score(regr_ada_2,feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday']],values_train, scoring="neg_mean_squared_error",cv=4)
tree_rmse_scores_test_ada_2 = np.sqrt(-scores_ada_2)


# In[219]:


display_scores(tree_rmse_scores_test_ada_2)


# In[70]:


y_train_predicted = regr_ada_2.predict(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
y_test_predicted_pruned_trees = regr_ada_2.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
mse_train = sqrt(mean_squared_error(values_train, y_train_predicted))
mse_test = sqrt(mean_squared_error(values_test, y_test_predicted_pruned_trees))
print("RF with pruned trees, Train MSE: {} Test MSE: {}".format(mse_train, mse_test))


# In[71]:


y_train_predicted_mape = regr_ada_2.predict(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
y_test_predicted_pruned_trees_mape = regr_ada_2.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
mape_train = mean_absolute_percentage_error(values_train, y_train_predicted_mape)
mape_test = mean_absolute_percentage_error(values_test, y_test_predicted_pruned_trees_mape)
print("RF with pruned trees, Train MAPE: {} Test MAPE: {}".format(mape_train, mape_test))


# In[ ]:





# In[40]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# In[1]:


#GRADIENT BOOSTING


# In[22]:


#Fazendo o GBRT COM PARAMETROS MAIS SIMPLES


# In[428]:


param_grid = {
    'max_depth': [1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50],
    'max_features': [None],
    'n_estimators': [200, 300, 500, 800, 1200, 1500, 1800]
}
# Create a based model
gb = GradientBoostingRegressor()
# Instantiate the grid search model
grid_search_gb = GridSearchCV(estimator = gb, param_grid = param_grid, 
                          cv = 4, n_jobs = -1, verbose = 2)


# In[429]:


grid_search_gb.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday', 'max_volume_weekday', 'min_volume_weekday', 'mediana_volume_weekday']], values_train)


# In[430]:


grid_search_gb.best_params_


# In[51]:



regressor_cubic_g = GradientBoostingRegressor(n_estimators=200,max_depth=3)


# In[52]:


regressor_cubic_g.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)
yhat = regressor_cubic_g.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])


# In[97]:


mean_absolute_percentage_error(values_test, yhat)


# In[98]:


rmse = sqrt(mean_squared_error(values_test, yhat))
rmse


# In[53]:


mean_absolute_error(values_test, yhat)


# In[99]:


regressor_cubic_g.score(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[100]:


regressor_cubic_g.score(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_test)


# In[101]:


regressor_cubic_g.feature_importances_


# In[379]:


feat_importances = pd.Series(regressor_cubic_g.feature_importances_, index=feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']].columns)
feat_importances.nlargest(25).plot(kind='barh')
plt.rcParams['figure.figsize'] = (11,8)
plt.xscale('log')


# In[102]:


denominator_gbrt = yhat.dot(yhat) - yhat.mean() * yhat.sum()


# In[103]:


m_gbrt = ( yhat.dot(values_test) - values_test.mean() * yhat.sum()) / denominator_gbrt

b_gbrt = ( values_test.mean() * yhat.dot(yhat) - yhat.mean() * yhat.dot(values_test)) / denominator_gbrt


# In[104]:


pred_gbrt = m_gbrt * yhat + b_gbrt


# In[105]:


plt.scatter(yhat, values_test)
plt.plot(yhat, pred_gbrt, 'r')


# In[106]:


res_gbrt = values_test - yhat
tot_gbrt = values_test - values_test.mean()


# In[107]:


R_squared_gbrt = 1 - res_gbrt.dot(res_gbrt) / tot_gbrt.dot(tot_gbrt)


# In[108]:


print(R_squared_gbrt)


# In[163]:


scores_gbrt = cross_val_score(regressor_cubic_g,feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']],values_train, scoring="neg_mean_squared_error",cv=4)
tree_rmse_scores_test_gbrt = np.sqrt(-scores_gbrt)


# In[164]:


display_scores(tree_rmse_scores_test_gbrt)


# In[54]:


scores_gbrt_mae = cross_val_score(regressor_cubic_g,feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']],values_train, scoring="neg_mean_absolute_error",cv=4)
tree_rmse_scores_test_gbrt_mae = np.sqrt(-scores_gbrt_mae)


# In[57]:


display_scores(tree_rmse_scores_test_gbrt_mae)


# In[109]:


y_train_predicted_gbrt = regressor_cubic_g.predict(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
y_test_predicted_pruned_trees_gbrt = regressor_cubic_g.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
rmse_train_gbrt = sqrt(mean_squared_error(values_train, y_train_predicted_gbrt))
rmse_test_gbrt = sqrt(mean_squared_error(values_test, y_test_predicted_pruned_trees_gbrt))
print("GBRT with pruned trees, Train RMSE: {} Test RMSE: {}".format(rmse_train_gbrt, rmse_test_gbrt))


# In[403]:


y_train_predicted_mape_gbrt = regressor_cubic_g.predict(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
y_test_predicted_mape_gbrt = regressor_cubic_g.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']])
mape_train_gbrt = mean_absolute_percentage_error(values_train, y_train_predicted_mape_gbrt)
mape_test_gbrt = mean_absolute_percentage_error(values_test, y_test_predicted_mape_gbrt)
print("GBRT, Train MAPE: {} Test MAPE: {}".format(mape_train_gbrt, mape_test_gbrt))


# In[402]:


title = "Learning Curves (GBRT)" 
estimator_gbrt = GradientBoostingRegressor(n_estimators=200,max_depth=3) 
plot_learning_curve(estimator_gbrt, title, feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train, cv=4, n_jobs=-1) 
plt.show()


# In[208]:


linear_regression = LinearRegression()


# In[209]:


linear_regression.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday']], values_train)


# In[210]:


linear = linear_regression.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday']])


# In[211]:


mean_absolute_percentage_error(values_test, linear)


# In[212]:


rmse = sqrt(mean_squared_error(values_test, linear))
rmse


# In[217]:


linear_regression.score(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday']], values_train,  sample_weight=None)


# In[216]:


linear_regression.score(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday']], values_test,  sample_weight=None)


# In[215]:


linear_regression.feature_importances_


# In[475]:


X_df = pd.DataFrame(data=feature_train)
X_df['LogMedHouseVal'] = values_train
_ = X_df.hist(column=['time_window', 'tollgate_id', 'direction'])


# In[479]:


X_df = pd.DataFrame(data=df_train)
X_df['LogMedHouseVal'] = values_train
_ = X_df.hist(column=['time_window', 'tollgate_id', 'direction'])


# In[496]:


scatter_plot = plt.scatter(df_train['volume'], df_train['hour'], alpha=0.5, 
                           c=df_train['volume'])
plt.show()


# In[497]:


df_train['volume'].fillna(df_train['volume'].mean(), inplace=True)
histogram_example = plt.hist(df_train['volume'], bins=15)
plt.show()


# In[207]:



# Criando um gráfico
plt.bar(df_train['tollgate_id'], df_train['volume'],  label = 'Barrar1', color = 'r')
plt.legend()
 
plt.show()


# In[192]:


scaler = StandardScaler()


# In[193]:


features_2 = scaler.fit_transform(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday']])


# In[194]:


regressor_cubic_g.fit(features_2, values_train)
yhat = regressor_cubic_g.predict(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday']])


# In[195]:


mean_absolute_percentage_error(values_test, y_pred_ada)


# In[196]:


rmse = sqrt(mean_squared_error(values_test, y_pred))
rmse


# In[200]:


regressor_cubic_g.score(features_2, values_train)


# In[201]:


features_test_2 = scaler.fit_transform(feature_test[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday']])


# In[203]:


regressor_cubic_g.score(features_test_2, values_test)


# In[204]:


regressor_cubic_g.feature_importances_


# In[564]:


result = []
test_data["volume"] = yhat
result.append(test_data[['tollgate_id', 'time_window', 'direction', 'volume']])


# In[565]:


df_result = pd.concat(result, axis=0)
df_result.to_csv("result/result_split_gbrt_"+str(np.mean(df_result["volume"]))+".csv", index=False)


# In[566]:


result_2 = []
test_data["volume"] = y_pred
result_2.append(test_data[['tollgate_id', 'time_window', 'direction', 'volume']])


# In[567]:


df_result_2 = pd.concat(result_2, axis=0)
df_result_2.to_csv("result/result_split_rf_"+str(np.mean(df_result["volume"]))+".csv", index=False)


# In[568]:


result_3 = []
test_data["volume"] = y_pred_ada
result_3.append(test_data[['tollgate_id', 'time_window', 'direction', 'volume']])


# In[569]:


df_result_3 = pd.concat(result_3, axis=0)
df_result_3.to_csv("result/result_split_ada_"+str(np.mean(df_result_3["volume"]))+".csv", index=False)


# In[570]:


result_rf = pd.read_csv('result/result_split_rf_73.29286067085843.csv')
result_gbrt = pd.read_csv('result/result_split_gbrt_73.29286067085843.csv')
result_ada = pd.read_csv('result/result_split_ada_74.17412982097916.csv')


# In[588]:


def Test2(rootDir):
    file_list = []
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        file_list.append(path)
        if os.path.isdir(path):
            Test2(path)
    return file_list

path = "result"
file_list = Test2(path)
df = pd.read_csv(file_list[0])

file_list.remove(file_list[0])
for x in file_list:
    dftmp = pd.read_csv(x)
    print(dftmp)
    df = df.merge(dftmp, on=["tollgate_id", "time_window", "direction"])

result_list = []
for index, row in df.iterrows():
    volume_list = row[3:].tolist()
    # print volume_list
    volume_list1 = sorted(volume_list)

    result = np.mean([volume_list1[0], volume_list1[1]])
    # result = np.mean(volume_list1)

    result_list.append(result)

df = df[["tollgate_id", "time_window", "direction"]]
df["volume"] = result_list

df["time_window_start"] = pd.to_datetime(df["time_window"])
df["time_window_end"] = df["time_window_start"] + timedelta(minutes=20)
list_tw = []
for x in range(0, len(df["time_window_start"] )):
    str_tw =  '[' + str(df["time_window_start"][x]) + ',' + str(df["time_window_end"][x]) + ')'
    list_tw.append(str_tw)

df["time_window"] = list_tw
df = df[["tollgate_id", "time_window", "direction","volume"]]


# In[585]:


del df


# In[586]:


df

