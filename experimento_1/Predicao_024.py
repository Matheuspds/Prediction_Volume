
# coding: utf-8

# In[1]:


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


# In[65]:


v_train = pd.read_csv('treino_agregado.csv')
v_test = pd.read_csv('teste_agregado.csv')


# In[66]:


#Adicionando valor da janela de tempo anterior na janela de tempo atual
v_train["volume_anterior"] = v_train.groupby(['direction', 'tollgate_id'])["volume"].transform("shift")
#condition = ~(df.groupby("time_window")['time'].transform("shift") == df['time'])
#df.loc[ condition,"volume_anterior" ] = None
v_train["volume_anterior"] =v_train.groupby(['direction', 'tollgate_id'])["volume_anterior"].fillna(method="ffill").fillna(0)


# In[67]:


#Adicionando valor da janela de tempo anterior na janela de tempo atual
v_train["volume_anterior_2"] = v_train.groupby(['direction', 'tollgate_id'])["volume"].transform("shift", 2)
#condition = ~(df.groupby("time_window")['time'].transform("shift") == df['time'])
#df.loc[ condition,"volume_anterior" ] = None
v_train["volume_anterior_2"] =v_train.groupby(['direction', 'tollgate_id'])["volume_anterior_2"].fillna(method="ffill").fillna(0)


# In[69]:


#Adicionando valor da janela de tempo anterior na janela de tempo atual
v_train["volume_proximo"] = v_train.groupby(['direction', 'tollgate_id'])["volume"].transform("shift", -1)
#condition = ~(df.groupby("time_window")['time'].transform("shift") == df['time'])
#df.loc[ condition,"volume_anterior" ] = None
v_train["volume_proximo"] =v_train.groupby(['direction', 'tollgate_id'])["volume_proximo"].fillna(method="ffill").fillna(0)


# In[71]:


#Adicionando valor da janela de tempo anterior na janela de tempo atual
v_train["volume_proximo_2"] = v_train.groupby(['direction', 'tollgate_id'])["volume"].transform("shift", -2)
#condition = ~(df.groupby("time_window")['time'].transform("shift") == df['time'])
#df.loc[ condition,"volume_anterior" ] = None
v_train["volume_proximo_2"] =v_train.groupby(['direction', 'tollgate_id'])["volume_proximo_2"].fillna(method="ffill").fillna(0)


# In[73]:


v_train['avg_vol_dia_semana'] = v_train.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean)


# In[75]:


v_train['desvio_padrao'] = v_train.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.std)


# In[77]:


v_train['desvio_padrao'].fillna(v_train.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)


# In[78]:


v_train


# In[79]:


v_train.to_csv('data_process_final/treino_final.csv', index=False)


# In[ ]:


#AGORA FAZER O MESMO COM OS DADOS DE TESTE


# In[80]:


#Adicionando valor da janela de tempo anterior na janela de tempo atual
v_test["volume_anterior"] = v_test.groupby(['direction', 'tollgate_id'])["volume"].transform("shift")
#condition = ~(df.groupby("time_window")['time'].transform("shift") == df['time'])
#df.loc[ condition,"volume_anterior" ] = None
v_test["volume_anterior"] =v_test.groupby(['direction', 'tollgate_id'])["volume_anterior"].fillna(method="ffill").fillna(0)


# In[81]:


#Adicionando valor da janela de tempo anterior na janela de tempo atual
v_test["volume_anterior_2"] = v_test.groupby(['direction', 'tollgate_id'])["volume"].transform("shift", 2)
#condition = ~(df.groupby("time_window")['time'].transform("shift") == df['time'])
#df.loc[ condition,"volume_anterior" ] = None
v_test["volume_anterior_2"] =v_test.groupby(['direction', 'tollgate_id'])["volume_anterior_2"].fillna(method="ffill").fillna(0)


# In[82]:


#Adicionando valor da janela de tempo anterior na janela de tempo atual
v_test["volume_proximo"] = v_test.groupby(['direction', 'tollgate_id'])["volume"].transform("shift", -1)
#condition = ~(df.groupby("time_window")['time'].transform("shift") == df['time'])
#df.loc[ condition,"volume_anterior" ] = None
v_test["volume_proximo"] =v_test.groupby(['direction', 'tollgate_id'])["volume_proximo"].fillna(method="ffill").fillna(0)


# In[83]:


#Adicionando valor da janela de tempo anterior na janela de tempo atual
v_test["volume_proximo_2"] = v_test.groupby(['direction', 'tollgate_id'])["volume"].transform("shift", -2)
#condition = ~(df.groupby("time_window")['time'].transform("shift") == df['time'])
#df.loc[ condition,"volume_anterior" ] = None
v_test["volume_proximo_2"] =v_test.groupby(['direction', 'tollgate_id'])["volume_proximo_2"].fillna(method="ffill").fillna(0)


# In[85]:


v_test['avg_vol_dia_semana'] = v_test.groupby(['time_window', 'direction', 'tollgate_id'])["volume"].transform(np.mean)


# In[86]:


v_test['desvio_padrao'] = v_test.groupby(['time_window', 'direction', 'tollgate_id'])["volume"].transform(np.std)


# In[87]:


v_test['desvio_padrao'].fillna(v_test.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)


# In[88]:


v_test


# In[89]:


v_test.to_csv('data_process_final/teste_final.csv', index=False)


# In[91]:


df_train = pd.read_csv('data_process_final/treino_final.csv')
df_teste = pd.read_csv('data_process_final/teste_final.csv')


# In[94]:


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


# In[95]:


feature_train, feature_test, values_train, values_test = feature_format()


# In[96]:


feature_train.head()


# In[97]:


rng = np.random.RandomState(1)
regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 50),
                         n_estimators=300, random_state = rng)


# In[119]:


regr.fit(feature_train[['tollgate_id', 'direction', 'week', 'weekend', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao']], values_train)

y_pred = regr.predict(feature_test[['tollgate_id', 'direction', 'week', 'weekend', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao']])


# In[99]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# In[120]:


mean_absolute_percentage_error(values_test, y_pred)


# In[122]:


rf = RandomForestRegressor(n_estimators = 1500, random_state = 42)


# In[123]:


rf.fit(feature_train[['tollgate_id', 'direction', 'week', 'weekend', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao']], values_train)


# In[124]:


# Use the forest's predict method on the test data
predictions = rf.predict(feature_test[['tollgate_id', 'direction', 'week', 'weekend', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao']])
# Calculate the absolute errors
errors = abs(predictions - values_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# In[125]:


# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / values_test)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')


# In[126]:


mean_absolute_percentage_error(values_test, predictions)

