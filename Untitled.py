
# coding: utf-8

# In[406]:


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
from sklearn.metrics import mean_squared_error
from math import sqrt


# In[286]:


pd_volume_train = pd.read_csv('processed_train_volume2.csv')
pd_volume_train.tail()


# In[275]:


#Função que será usada para obter a janela de tempo de 5 minutos
def get_timewindow(t):
        time_window = 5
        if t.minute < time_window:
            window = [time(t.hour, 0), time(t.hour,5)]
        elif t.minute < time_window*2:
            window = [time(t.hour, 5), time(t.hour, 10)]
        elif t.minute < time_window*3:
            window = [time(t.hour, 10), time(t.hour, 15)]
        elif t.minute < time_window*4:
            window = [time(t.hour, 15), time(t.hour, 20)]
        elif t.minute < time_window*5:
            window = [time(t.hour, 20), time(t.hour, 25)]
        elif t.minute < time_window*6:
            window = [time(t.hour, 25), time(t.hour, 30)]
        elif t.minute < time_window*7:
            window = [time(t.hour, 30), time(t.hour, 35)]
        elif t.minute < time_window*8:
            window = [time(t.hour, 35), time(t.hour, 40)]
        elif t.minute < time_window*9:
            window = [time(t.hour, 40), time(t.hour, 45)]
        elif t.minute < time_window*10:
            window = [time(t.hour, 45), time(t.hour, 50)]
        elif t.minute < time_window*11:
            window = [time(t.hour, 50), time(t.hour, 55)]
        else:
            try:
                window = [time(t.hour, 55), time(t.hour + 1, 0)]
            except ValueError:
                window = [time(t.hour, 55), time(0,0,0)]
        s_window = '[' + str(window[0]) + ',' + str(window[1]) + ')'
        return s_window

def get_hour(t):
        return t.hour


# In[124]:


#Função que será usada para obter a janela de tempo de 20 minutos
def get_timewindow20(t):
        time_window = 20
        if t.minute < time_window:
            window = [time(t.hour, 0), time(t.hour,20)]
        elif t.minute < time_window*2:
            window = [time(t.hour, 20), time(t.hour, 40)]
        else:
            try:
                window = [time(t.hour, 40), time(t.hour + 1, 0)]
            except ValueError:
                window = [time(t.hour, 40), time(0,0,0)]
        s_window = '[' + str(window[0]) + ',' + str(window[1]) + ')'
        return s_window

def get_hour(t):
        return t.hour


# In[287]:


del pd_volume_train['time_window']
pd_volume_train['time'] = pd.to_datetime(pd_volume_train['time'], format = '%Y-%m-%d %H:%M:%S')
pd_volume_train['t'] = pd_volume_train['time'].dt.time
pd_volume_train['time_window'] = pd_volume_train['t'].apply(get_timewindow)
#del pd_volume_train['t']
pd_volume_train.head()


# In[288]:


pd_volume_train['time'] =  pd.to_datetime(pd_volume_train['time'] , format='%Y-%m-%d %H:%M:%S')
#pd_volume_train = pd_volume_train.set_index(['time_window'])

# 车流量
pd_volume_train = pd_volume_train.groupby([pd.Grouper(freq='5T', key='time'), 'tollgate_id', 'direction', 'time_window', 'date', 'hour', 'holiday']).size()       .reset_index().rename(columns = {0:'volume'})


# In[289]:


pd_volume_train.head()


# In[290]:


pd_volume_train['weekday'] = pd_volume_train['time'].dt.dayofweek + 1


# In[291]:


#Adicionando valor da janela de tempo (2x) anterior na janela de tempo atual
pd_volume_train["volume_anterior"] = pd_volume_train.groupby(['direction', 'tollgate_id'])["volume"].transform("shift")
#condition = ~(df.groupby("time_window")['time'].transform("shift") == df['time'])
#df.loc[ condition,"volume_anterior" ] = None
pd_volume_train["volume_anterior"] =pd_volume_train.groupby(['direction', 'tollgate_id'])["volume_anterior"].fillna(method="ffill").fillna(0)
pd_volume_train.head()


# In[293]:


#Adicionando valor da janela de tempo (2x) anterior na janela de tempo atual
pd_volume_train["volume_anterior_2"] = pd_volume_train.groupby(['direction', 'tollgate_id'])["volume"].transform("shift", 2)
#condition = ~(df.groupby("time_window")['time'].transform("shift") == df['time'])
#df.loc[ condition,"volume_anterior" ] = None
pd_volume_train["volume_anterior_2"] =pd_volume_train.groupby(['direction', 'tollgate_id'])["volume_anterior_2"].fillna(method="ffill").fillna(0)
pd_volume_train.head()


# In[294]:


pd_volume_train['media_weekday'] = pd_volume_train.groupby(['time_window', 'weekday', 'direction', 'tollgate_id'])["volume"].transform(np.mean)


# In[295]:


pd_volume_train['desvio_weekady'] = pd_volume_train.groupby(['time_window', 'weekday', 'direction', 'tollgate_id'])["volume"].transform(np.std)


# In[296]:


pd_volume_train.isnull().sum()


# In[297]:


pd_volume_train['desvio_weekady'].fillna(pd_volume_train.groupby(['time_window', 'weekday', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
pd_volume_train.isnull().sum()


# In[338]:


pd_volume_train


# In[353]:


pd_volume_train['media_dia_hora'] = pd_volume_train.groupby(['date', 'hour', 'tollgate_id', 'direction'])["volume"].transform(np.mean)


# In[354]:


pd_volume_train


# In[355]:


pd_volume_train['desvio_dia_hora'] = pd_volume_train.groupby(['date', 'hour', 'direction', 'tollgate_id'])["volume"].transform(np.std)


pd_volume_train['desvio_dia_hora'].fillna(pd_volume_train.groupby(['date', 'hour', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
pd_volume_train.isnull().sum()


# #SALVANDO ARQUIVO DE TREINO INICIANDO MESMAS MODIFICAÇÕES PARA ARQUIVO DE TESTE

# In[356]:


pd_volume_train.to_csv('dados_treino_volume_5Min.csv', index = False)


# In[367]:


pd_volume_test = pd.read_csv('processed_test_volume2.csv')
pd_volume_test.head()
del pd_volume_test['time_window']


# In[368]:


pd_volume_test['time'] = pd.to_datetime(pd_volume_test['time'], format = '%Y-%m-%d %H:%M:%S')
pd_volume_test['t'] = pd_volume_test['time'].dt.time
pd_volume_test['time_window'] = pd_volume_test['t'].apply(get_timewindow20)
#del pd_volume_train['t']
pd_volume_test.tail()


# In[369]:


pd_volume_test['time'] =  pd.to_datetime(pd_volume_test['time'] , format='%Y-%m-%d %H:%M:%S')
#pd_volume_train = pd_volume_train.set_index(['time_window'])

# 车流量
pd_volume_test = pd_volume_test.groupby([pd.Grouper(freq='20T', key='time'), 'tollgate_id', 'direction', 'time_window', 'date', 'hour', 'holiday']).size()       .reset_index().rename(columns = {0:'volume'})


# In[370]:


pd_volume_test['weekday'] = pd_volume_test['time'].dt.dayofweek + 1


# In[371]:


#Adicionando valor da janela de tempo (2x) anterior na janela de tempo atual
pd_volume_test["volume_anterior"] = pd_volume_test.groupby(['direction', 'tollgate_id'])["volume"].transform("shift")
#condition = ~(df.groupby("time_window")['time'].transform("shift") == df['time'])
#df.loc[ condition,"volume_anterior" ] = None
pd_volume_test["volume_anterior"] =pd_volume_test.groupby(['direction', 'tollgate_id'])["volume_anterior"].fillna(method="ffill").fillna(0)
pd_volume_test.head()


# In[372]:


#Adicionando valor da janela de tempo (2x) anterior na janela de tempo atual
pd_volume_test["volume_anterior_2"] = pd_volume_test.groupby(['direction', 'tollgate_id'])["volume"].transform("shift", 2)
#condition = ~(df.groupby("time_window")['time'].transform("shift") == df['time'])
#df.loc[ condition,"volume_anterior" ] = None
pd_volume_test["volume_anterior_2"] =pd_volume_test.groupby(['direction', 'tollgate_id'])["volume_anterior_2"].fillna(method="ffill").fillna(0)
pd_volume_test.head()


# In[373]:


pd_volume_test['media_weekday'] = pd_volume_test.groupby(['time_window', 'direction', 'tollgate_id'])["volume"].transform(np.mean)


# In[374]:


pd_volume_test['desvio_weekady'] = pd_volume_test.groupby(['time_window', 'direction', 'tollgate_id'])["volume"].transform(np.std)


# In[375]:


pd_volume_test.tail()


# In[376]:


#pd_volume_train['desvio_weekady'].fillna(pd_volume_train.groupby(['time_window', 'weekday', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
pd_volume_test.isnull().sum()


# In[377]:


pd_volume_test['media_dia_hora'] = pd_volume_test.groupby(['date', 'hour', 'direction', 'tollgate_id'])["volume"].transform(np.mean)


# In[378]:


pd_volume_test['desvio_dia_hora'] = pd_volume_test.groupby(['date', 'hour', 'direction', 'tollgate_id'])["volume"].transform(np.std)


pd_volume_test['desvio_dia_hora'].fillna(pd_volume_test.groupby(['date', 'hour', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
pd_volume_test.isnull().sum()


# In[379]:


pd_volume_test.to_csv('dados_teste_volume_5Min.csv', index = False)


# In[390]:


def feature_format():
    v_train = pd.read_csv('dados_treino_volume_5Min.csv')
    v_test = pd.read_csv('dados_teste_volume_5Min.csv')
    #v_train = v_train.set_index(['time'])
    #v_test = v_test.set_index(['time'])
    #volume_train = v_train.groupby(['time_window','tollgate_id','direction','date', 'hour', 'has_etc', 'holiday']).size().reset_index().rename(columns = {0:'volume'})
    #volume_test = v_test.groupby(['time_window','tollgate_id','direction','date', 'hour', 'has_etc', 'holiday']).size().reset_index().rename(columns = {0:'volume'})
    #print(volume_train)                
    x = pd.Series(v_train['time_window'].unique())
    xT = pd.Series(v_test['time_window'].unique())
    s = pd.Series(range(len(x)),index = x.values)
    sT = pd.Series(range(len(xT)),index = xT.values)
    v_train['window_n'] = v_train['time_window'].map(s)
    v_test['window_n'] = v_test['time_window'].map(sT)
#        print vol_test.tail()
    #volume_train['weekday'] = v_train['weekday']
    #volume_test['weekday'] = v_test['weekday']
    
    feature_train = v_train.drop('volume', axis = 1)
    feature_test = v_test.drop('volume',axis = 1)
    values_train = v_train['volume'].values
    values_test = v_test['volume'].values
    
    return feature_train, feature_test, values_train, values_test


# In[391]:


feature_train, feature_test, values_train, values_test = feature_format()


# In[399]:


values_test


# In[416]:


rng = np.random.RandomState(1)
regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 100),
                         n_estimators=50, random_state = rng)


# In[415]:


regr.fit(feature_train[['window_n','tollgate_id', 'direction','media_weekday','weekday', 'desvio_weekady', 'media_dia_hora', 'desvio_dia_hora', 'volume_anterior']], values_train)

y_pred = regr.predict(feature_test[['window_n','tollgate_id', 'direction', 'media_weekday', 'weekday', 'desvio_weekady', 'media_dia_hora', 'desvio_dia_hora', 'volume_anterior']])

mape = np.mean(np.abs((y_pred - values_test)/values_test))
mape


# In[333]:


#Função que calcula o MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[334]:


mean_absolute_percentage_error(y_pred, values_test)


# In[417]:


# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1500, random_state = 42)
# Train the model on training data
rf.fit(feature_train[['window_n','tollgate_id', 'direction', 'media_weekday', 'weekday', 'desvio_weekady', 'media_dia_hora', 'desvio_dia_hora', 'volume_anterior']], values_train);


# In[418]:


# Use the forest's predict method on the test data
predictions = rf.predict(feature_test[['window_n','tollgate_id', 'direction', 'media_weekday', 'weekday', 'desvio_weekady', 'media_dia_hora', 'desvio_dia_hora', 'volume_anterior']])
# Calculate the absolute errors
errors = abs(predictions - values_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[419]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / values_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[407]:


rmse = sqrt(mean_squared_error(values_test, y_pred))
rmse

