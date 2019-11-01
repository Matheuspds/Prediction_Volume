
# coding: utf-8

# In[448]:


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from datetime import time
import matplotlib.pyplot as pplot
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import math


# In[2]:


# Descrição das features:
# time           datetime  Data e Hora em que o veículo passa pelo pedágio;
# tollgate_id    string    Identificador do pedágio;
# direction      string    0: entra na rodovia pelo pedágio; 1: sai da rodovia pelo pedágio;
# vehicle_model  int       Um número que indica a capacidade do veículo;
# has_etc        string    Indica se o veículo possui ou não o sistema ETC; 0 - NÃO, 1 - SIM
# vehicle_type   string    0: veículo de passageiro; 1: veículo de carga
# weekday        int       Representa os dias da semana
# weekend        int       1: Para quando for fim de semana; 0: Para quando não for fim de semana


# In[476]:


pd_volume_train = pd.read_csv('processed_train_volume2.csv')
#pd_volume_test = pd.read_csv('processed_test_volume2.csv')


# In[477]:


pd_volume_train.head()
#pd_volume_test.head()


# In[478]:


pd_volume_train['time'] =  pd.to_datetime(pd_volume_train['time'] , format='%Y-%m-%d %H:%M:%S')
#pd_volume_train = pd_volume_train.set_index(['time_window'])

# 车流量
pd_volume_train = pd_volume_train.groupby([pd.Grouper(freq='20T', key='time'), 'tollgate_id', 'direction', 'time_window', 'date', 'hour']).size()       .reset_index().rename(columns = {0:'volume'})


# In[479]:


pd_volume_train.head()


# In[480]:


pd_volume_train['weekday'] = pd_volume_train['time'].dt.dayofweek + 1


# In[481]:


pd_volume_train[pd_volume_train['weekday'] == 3].head()


# In[482]:


pd.get_dummies(pd_volume_train['time_window']).head()


# In[50]:


# Converte array em matriz
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# In[483]:


# Transpõe
#df_transp = df.T

# Seta o nome das colunas para os valores da primeira linha
#df_transp.columns = df_transp.iloc[0]

# Define os dados começando a partir da segunda linha
#df_transp = df_transp[1:]

#df_transp.loc[:,'media'] = df_transp.mean(numeric_only=True, axis=0).values

#df_transf = pd_volume_train.set_index(['time'])

# 车流量
#t6_train = t6_train.groupby([pd.TimeGrouper('20Min'), 'tollgate_id', 'direction']).size()\
#       .reset_index().rename(columns = {0:'volume'})

#t6_train = t6_train.set_index(['time'])

# 车流量
#t6_train = t6_train.groupby([pd.TimeGrouper('20Min'), 'tollgate_id', 'direction']).size()\
#       .reset_index().rename(columns = {0:'volume'})

#df_transf = pd_volume_train.groupby(['time_window','weekday','tollgate_id', 'direction', 'date']).size()\
 #      .reset_index().rename(columns = {0:'volume'})

df_transf = pd_volume_train
df_transf.head()


# In[484]:


df_transf['media_volume'] = df_transf.groupby(['time_window', 'weekday', 'direction', 'tollgate_id'])["volume"].transform(np.mean)


# In[485]:


#df_transf['media_volume'] = df_transf['soma'].mean
df_transf.head()
#del df_transf['soma']


# In[379]:


values = np.array(df_transf['volume'])


# In[382]:


values


# In[486]:


df_transf['desvio_padrao'] = df_transf.groupby(['time_window', 'weekday', 'direction', 'tollgate_id'])["volume"].transform(np.std)


# In[487]:


df_transf.head()
#del df_transf['desvio_padrao']


# In[489]:


#df_transf['mediaArredondada'] = df_transf['media_volume'].sum()
#df_transf['mediaArrendodada'] = df_transf['media_volume'].mean()
df_transf['desvio_padrao'].fillna(df_transf.groupby(['time_window', 'weekday', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
#del df_transf['mediaArredondada']


# In[490]:


df_transf.isnull().sum()


# In[491]:


df_transf.to_csv('dados_com_media_desvio_treino_volume.csv', index = False)


# In[446]:


#df_transf.isnull().sum()
medi


# In[407]:


pd_volume_test = pd.read_csv('processed_test_volume2.csv')


# In[408]:


pd_volume_test.head()


# In[409]:


pd_volume_test['time'] =  pd.to_datetime(pd_volume_test['time'] , format='%Y-%m-%d %H:%M:%S')
#pd_volume_train = pd_volume_train.set_index(['time_window'])

# 车流量
pd_volume_test = pd_volume_test.groupby([pd.Grouper(freq='20T', key='time'), 'tollgate_id', 'direction', 'time_window', 'date', 'hour']).size()       .reset_index().rename(columns = {0:'volume'})


# In[410]:


pd_volume_test.head()


# In[412]:


pd_volume_test['weekday'] = pd_volume_test['time'].dt.dayofweek + 1


# In[414]:


pd_volume_test[pd_volume_test['weekday'] == 3].head()


# In[423]:


pd_volume_test['media_volume'] = pd_volume_test.groupby(['time_window', 'direction', 'tollgate_id'])["volume"].transform(np.mean)


# In[424]:


pd_volume_test.head()


# In[425]:


pd_volume_test['desvio_padrao'] = pd_volume_test.groupby(['time_window', 'direction', 'tollgate_id'])["volume"].transform(np.std)


# In[444]:


pd_volume_test.isnull().sum()


# In[427]:


pd_volume_test.to_csv('dados_com_media_desvio_sem_weekday_treino_volume.csv', index = False)


# In[492]:


def feature_format():
    v_train = pd.read_csv('dados_com_media_desvio_treino_volume.csv')
    v_test = pd.read_csv('dados_com_media_desvio_sem_weekday_treino_volume.csv')
    #pd_volume_train = pd_volume_train.set_index(['time'])
    #pd_volume_test = pd_volume_test.set_index(['time'])
    #volume_train = v_train.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #volume_test = v_test.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #print(volume_train)                
    x = pd.Series(v_train['time_window'].unique())
    s = pd.Series(range(len(x)),index = x.values)
    v_train['window_n'] = v_train['time_window'].map(s)
    v_test['window_n'] = v_test['time_window'].map(s)
#        print vol_test.tail()
    #volume_train['weekday'] = v_train['weekday']
    #volume_test['weekday'] = v_test['weekday']
    
    feature_train = v_train.drop('volume', axis = 1)
    feature_test = v_test.drop('volume',axis = 1)
    values_train = v_train['volume'].values
    values_test = v_test['volume'].values
    
    return feature_train, feature_test, values_train, values_test


# In[493]:


feature_train, feature_test, values_train, values_test = feature_format()


# In[494]:


feature_train.isnull().sum()
#pd_volume_train[pd_volume_train['weekday'] == 3].head()


# In[495]:


values_train


# In[496]:


rng = np.random.RandomState(1)
regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 15),
                         n_estimators=20, random_state = rng)


# In[501]:


regr.fit(feature_train[['window_n', 'tollgate_id', 'direction', 'weekday', 'media_volume', 'desvio_padrao']], values_train)

y_pred = regr.predict(feature_test[['window_n', 'tollgate_id', 'direction', 'weekday', 'media_volume', 'desvio_padrao']])

mape = np.mean(np.abs((y_pred - values_test)/values_test))

#print (feature_test)
print (mape)


# In[500]:


#Função que calcula o MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[335]:


mean_absolute_percentage_error(y_pred, values_test)

