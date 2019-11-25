
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import random
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
path_final = 'data_final/'


# In[4]:


df_test1 = pd.read_csv(path_final+"test_final_0.csv")
df_test2 = pd.read_csv(path_final+"test_final_5.csv")
df_test3 = pd.read_csv(path_final+"test_final_10.csv")
df_test4 = pd.read_csv(path_final+"test_final_15.csv")
df_train1 = pd.read_csv(path_final+"train_final_0.csv")
df_train2 = pd.read_csv(path_final+"train_final_5.csv")
df_train3 = pd.read_csv(path_final+"train_final_10.csv")
df_train4 = pd.read_csv(path_final+"train_final_15.csv")


# In[5]:


df_train_list = [df_train1,df_train2, df_train3, df_train4]
df_test_list = [df_test1, df_test2, df_test3, df_test4]


# In[7]:


df_train = pd.concat(df_train_list)


# In[8]:


df_train[(df_train['date'] == '2016-10-01') & (df_train['hour'] == 8)].head()


# In[9]:


df_remove = df_train.loc[(df_train['date'] >= '2016-10-01') 
                         & (df_train['date'] <= '2016-10-07') 
                        ]

df_train = df_train.drop(df_remove.index)


# In[10]:


df_train[(df_train['date'] == '2016-10-01') & (df_train['hour'] == 8)].head()


# In[11]:


df_test = pd.concat(df_test_list)


# In[12]:


df_test.head()


# In[13]:


df_remove = df_test.loc[(df_test['volume'] == 0)]

df_test = df_test.drop(df_remove.index)


# In[32]:


df_test.head()
df_test.to_csv("test_real.csv", index=False)


# In[14]:


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


# In[15]:


feature_train, feature_test, values_train, values_test = feature_format()
feature_train.head()


# In[19]:


def feature_transform_split(data):
    data["precipitation"] = data[["precipitation"]].fillna(value=0)
    data["rel_humidity"] = data[["rel_humidity"]].fillna(value=50)

    data["precipitation"] = data["precipitation"].apply(lambda x: x > 0)
    data["rel_humidity"] = data["rel_humidity"].apply(lambda x: x > 90)

    data = data.drop("precipitation", axis=1)
    return data


# In[47]:


def feature_transform_split_complete(key, data):
    # data = remove_exception(data)

    data["precipitation"] = data[["precipitation"]].fillna(value=0)
    data["rel_humidity"] = data[["rel_humidity"]].fillna(value=50)

    data["precipitation"] = data["precipitation"].apply(lambda x: x > 0)
    data["rel_humidity"] = data["rel_humidity"].apply(lambda x: x > 90)

    data = data.drop("precipitation", axis=1)
    # data = data.drop("rel_humidity", axis= 1)




    # data["sum"] = data["0"] + data["1"] + data["2"] + data["3"] + data["4"] + data["5"]

    #data = pd.concat([data, pd.get_dummies(data['period_num'])], axis=1)
    #data = data.drop("period_num", axis=1)

   # data = pd.concat([data, pd.get_dummies(data['holiday'])], axis=1)
    #data = data.drop("holiday", axis=1)
    #
    # data = pd.concat([data, pd.get_dummies(data['first_last_workday'])], axis=1)
    #data = data.drop("first_last_workday", axis=1)

    #data = data.drop("day_of_week", axis=1)

    if (key == 1):
        data = pd.concat([data, pd.get_dummies(data['tollgate_id'])], axis=1)
        # data["tollgate_id1"] = data['tollgate_id']
        data["direction1"] = data['direction']
    return data


# In[15]:


feature_train = feature_transform_split(feature_train)


# In[16]:


feature_test = feature_transform_split(feature_test)


# In[16]:


feature_train.head()


# In[17]:


regressor_cubic = RandomForestRegressor(n_estimators = 1500, random_state = 42)


# In[20]:


regressor_cubic.fit(feature_train[['weekday','volume_anterior', 'volume_anterior_2','media_volume', 'desvio_padrao', 'am_pm']], values_train)


# In[42]:


y_pred = regressor_cubic.predict(feature_test[['weekday','volume_anterior', 'volume_anterior_2','media_volume', 'desvio_padrao', 'am_pm']])y_pred = regressor_cubic.predict(feature_test[['weekday','volume_anterior', 'volume_anterior_2','media_volume', 'desvio_padrao', 'am_pm']])


# In[83]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[84]:


mean_absolute_percentage_error(values_test, y_pred)


# In[36]:


y_pred.transform("shift",1)
#pd_volume_test["volume_anterior_2"] = pd_volume_test.groupby(['direction', 'tollgate_id'])["volume"].transform("shift", 2)


# In[24]:


values_test


# In[43]:


y_pred = np.delete(y_pred, 0)
print (list(y_pred)[0])


# In[74]:


y_pred[[len(y_pred+1)]] = 2


# In[70]:


len(y_pred)+1


# In[77]:


values_test = np.delete(values_test, len(values_test)-1)


# In[78]:


print (list(values_test)[4]) 


# In[79]:


mean_squared_error(y_pred,values_test)


# In[80]:


r2_score(values_test, y_pred) 


# In[81]:


rms = sqrt(mean_squared_error(values_test, y_pred))
rms


# In[85]:


values_test


# In[86]:


y_pred

