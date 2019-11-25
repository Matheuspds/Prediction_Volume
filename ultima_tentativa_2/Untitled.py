
# coding: utf-8

# In[102]:


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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.cross_validation import ShuffleSplit, train_test_split
from sklearn.metrics import mean_squared_error
path_final = 'dados_finais/'
from sklearn import preprocessing
from sklearn import svm


# In[103]:


df_test1 = pd.read_csv(path_final+"test1.csv")
df_test2 = pd.read_csv(path_final+"test2.csv")
df_test3 = pd.read_csv(path_final+"test3.csv")
df_test4 = pd.read_csv(path_final+"test4.csv")
df_train1 = pd.read_csv(path_final+"train1.csv")
df_train2 = pd.read_csv(path_final+"train2.csv")
df_train3 = pd.read_csv(path_final+"train3.csv")
df_train4 = pd.read_csv(path_final+"train4.csv")


# In[104]:


def adiciona_media_desvio_por_dia(df):
    df['media_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])["volume"].transform(np.mean)
    df['desvio_padrao_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])["volume"].transform(np.std)
    df['desvio_padrao_hora_dia'].fillna(df.groupby(['direction', 'tollgate_id', 'day', 'hour'])["volume"].transform(np.mean), inplace=True)
    df['min_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])['volume'].transform(np.min)
    df['max_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])['volume'].transform(np.max)
    df['mediana_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])['volume'].transform(np.median)
    return df


# In[105]:


df_train1 = adiciona_media_desvio_por_dia(df_train1)
df_train2 = adiciona_media_desvio_por_dia(df_train2)
df_train3 = adiciona_media_desvio_por_dia(df_train3)
df_train4 = adiciona_media_desvio_por_dia(df_train4)
df_test1 = adiciona_media_desvio_por_dia(df_test1)


# In[106]:


df_train1.head()


# In[107]:


df_train_list = [df_train1,df_train2, df_train3, df_train4]
df_test_list = [df_test1, df_test2, df_test3, df_test4]


# In[108]:


random.shuffle(df_train_list)
df_train = pd.concat(df_train_list)


# In[109]:


df_train[(df_train['date'] == '2016-10-10') & (df_train['hour'] == 8)].head()


# In[110]:


df_test = df_test1


# In[111]:


df_remove = df_test.loc[(df_test['volume'] == 0)]

df_test = df_test.drop(df_remove.index)


# In[112]:


df_test.head()


# In[113]:


def feature_format():
    #pd_volume_train = pd_volume_train.set_index(['time'])
    #pd_volume_test = pd_volume_test.set_index(['time'])
    #volume_train = v_train.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #volume_test = v_test.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #print(volume_train)                
    x = pd.Series(df_train['time_window'].unique())
    s = pd.Series(range(len(x)),index = x.values)
    df_train['window_n'] = df_train['time_window'].map(s)
    df_test['window_n'] = df_test['time_window'].map(s)
#        print vol_test.tail()
    #volume_train['weekday'] = v_train['weekday']
    #volume_test['weekday'] = v_test['weekday']
    
    feature_train = df_train.drop('volume', axis = 1)
    feature_test = df_test.drop('volume',axis = 1)
    values_train = df_train['volume'].values
    values_test = df_test['volume'].values
    
    return feature_train, feature_test, values_train, values_test


# In[123]:


feature_train, feature_test, values_train, values_test = feature_format()
feature_train = pd.concat([feature_train, pd.get_dummies(feature_train['tollgate_id'])], axis=1)
feature_test = pd.concat([feature_test, pd.get_dummies(feature_test['tollgate_id'])], axis=1)


# In[118]:


feature_train.columns


# In[143]:


regressor_cubic = RandomForestRegressor(n_estimators = 1500, max_depth=10, max_features=None)


# In[157]:


regressor_cubic.fit(feature_train[['tollgate_id','direction', 'hour','weekday', 'weekend','volume_anterior', 'volume_anterior_2', 'desvio_padrao', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia']], values_train)


# In[158]:


y_pred = regressor_cubic.predict(feature_test[[1,2,3,'direction', 'hour','weekday', 'weekend','volume_anterior', 'volume_anterior_2', 'desvio_padrao', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia']])


# In[154]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# In[159]:


mean_absolute_percentage_error(values_test, y_pred)


# In[151]:


sqrt(mean_squared_error(y_pred, values_test))


# In[45]:


np.count_nonzero(values_test)


# In[47]:


df_train_grouped = df_train.groupby(["tollgate_id", "direction"])
df_test_grouped = df_test.groupby(["tollgate_id", "direction"])
result = []
oob = []
for key, train_data in df_train_grouped:
    test_data = df_test_grouped.get_group(key)
    len_train = len(train_data)
    train_data = train_data.append(test_data)[train_data.columns.tolist()]
    #train_data = feature_transform_split(key, train_data)
    train_data.head()

    regressor_cubic = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=10, oob_score=True)

    train_data = pd.DataFrame.reset_index(train_data)
    train_data = train_data.drop("index", axis=1)
    y = train_data.ix[:len_train - 1, :]["volume"]
    
    #del tain_data['date']
    #del train_data['time_start']
    #del train['time_window']
    #del train['minute']
    
    x = train_data.ix[:len_train - 1, :]
    x1 = train_data.ix[len_train:, :]
    del x['time_window']
    del x1['time_window']
    del x['minute']
    del x1['minute']
    del x['time_start']
    del x1['time_start']
    del x['date']
    del x1['date']
    regressor_cubic.fit(x, y)
    resultado_obtido = regressor_cubic.predict(x1)

    #df_h = test_data
    #df_h.to_csv("result/result_split_rf_huehue"+"teste"+".csv", index=False)

    test_data["volume"] = resultado_obtido
    result.append(test_data[['tollgate_id', 'time_window', 'direction', 'volume']])


# In[48]:


df_train_grouped


# In[121]:


df_result = pd.concat(result, axis=0)


# In[122]:


df_result.head()


# In[123]:


df_test.head()


# In[124]:


volumes_predict = df_result['volume'].values.astype(int)
volumes_real = df_test['volume'].values


# In[125]:


volumes_real


# In[126]:


volumes_predict


# In[127]:


mean_absolute_percentage_error(volumes_real, volumes_predict)


# In[128]:


sqrt(mean_squared_error(volumes_predict, volumes_real))


# In[74]:


for shift_num in range(0, 6):
        f2 = lambda x: x.values[shift_num]

        df_train1[str(shift_num)] = df_train1[["tollgate_id", "direction", "volume", "date", "am_pm"]].groupby(
            ["tollgate_id", "direction", "date", "am_pm"]).transform(f2)


# In[77]:


df_train1


# In[4]:


s = np.random.normal(100, 300, 50)


# In[5]:


s

