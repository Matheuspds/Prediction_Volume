
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import random


# In[20]:


df_test = pd.read_csv("test2.csv")

df_train0 = pd.read_csv("train.csv")
df_train1 = pd.read_csv("train1.csv")
df_train2 = pd.read_csv("train2.csv")
df_train3 = pd.read_csv("train3.csv")
df_train_list = [df_train0, df_train1, df_train2, df_train3]


# In[21]:


def feature_transform_split(key, data):
    # data = remove_exception(data)

    data["precipitation"] = data[["precipitation"]].fillna(value=0)
    data["rel_humidity"] = data[["rel_humidity"]].fillna(value=50)

    data["precipitation"] = data["precipitation"].apply(lambda x: x > 0)
    data["rel_humidity"] = data["rel_humidity"].apply(lambda x: x > 90)

    data = data.drop("precipitation", axis=1)
    # data = data.drop("rel_humidity", axis= 1)


    # data["sum"] = data["0"] + data["1"] + data["2"] + data["3"] + data["4"] + data["5"]

    data = pd.concat([data, pd.get_dummies(data['period_num'])], axis=1)
    data = data.drop("period_num", axis=1)

    data = pd.concat([data, pd.get_dummies(data['holiday'])], axis=1)
    data = data.drop("holiday", axis=1)
    #
    # data = pd.concat([data, pd.get_dummies(data['first_last_workday'])], axis=1)
    data = data.drop("first_last_workday", axis=1)

    data = data.drop("day_of_week", axis=1)

    if (key == 1):
        data = pd.concat([data, pd.get_dummies(data['tollgate_id'])], axis=1)
        # data["tollgate_id1"] = data['tollgate_id']
        data["direction1"] = data['direction']
    return data


# In[22]:


random.shuffle(df_train_list)
df_train = pd.concat(df_train_list)

#df_ts = pd.read_csv("ts_feature2_simple.csv")
df_date = pd.read_csv("date.csv")
df_train = df_train.merge(df_date, on="date", how="left")
#df_train = df_train.merge(df_ts, on=["tollgate_id", "hour", "miniute", "direction"], how="left")
df_test = df_test.merge(df_date, on="date", how="left")
#df_test = df_test.merge(df_ts, on=["tollgate_id", "hour", "miniute", "direction"], how="left")

df_train_grouped = df_train.groupby(["tollgate_id", "direction"])
df_test_grouped = df_test.groupby(["tollgate_id", "direction"])
df_train_grouped = df_train.groupby(["tollgate_id", "direction"])
df_test_grouped = df_test.groupby(["tollgate_id", "direction"])
result = []
oob = []
for key, train_data in df_train_grouped:
    test_data = df_test_grouped.get_group(key)
    len_train = len(train_data)
    train_data = train_data.append(test_data)[train_data.columns.tolist()]
    train_data = feature_transform_split(key, train_data)

    regressor_cubic = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=10, oob_score=True)

    train_data = pd.DataFrame.reset_index(train_data)
    train_data = train_data.drop("index", axis=1)
    y = train_data.ix[:len_train - 1, :]["volume"]

    x = train_data.ix[:len_train - 1, 8:]
    x1 = train_data.ix[len_train:, 8:]
    regressor_cubic.fit(x, y)
    yhat = regressor_cubic.predict(x1)
    
    test_data["volume"] = yhat
    result.append(test_data[['tollgate_id', 'time_window', 'direction', 'volume']])


# In[23]:


df_result = pd.concat(result, axis=0)

df_result.to_csv("result/result_split_rf_TESTAR_AGORA"+".csv", index=False)


# In[9]:


#regressor = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=10, oob_score=True)


# In[16]:


#regressor.fit(x, y)


# In[24]:


df_pred = pd.read_csv("result/result_split_rf_TESTAR_AGORA"+".csv")
df_real = pd.read_csv("resultado_real_teste.csv")


# In[25]:


df_pred.head()


# In[26]:


df_real.head()


# In[75]:


df_test_v = pd.read_csv("test2_no_filter.csv")
df_train_v = pd.read_csv("train_no_filter.csv")


# In[76]:


def feature_format():
    #pd_volume_train = pd_volume_train.set_index(['time'])
    #pd_volume_test = pd_volume_test.set_index(['time'])
    #volume_train = v_train.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #volume_test = v_test.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #print(volume_train)                
    x = pd.Series(df_train_v['time_window'].unique())
    s = pd.Series(range(len(x)),index = x.values)
    df_train_v['window_n'] = df_train_v['time_window'].map(s)
    df_test_v['window_n'] = df_test_v['time_window'].map(s)
#        print vol_test.tail()
    #volume_train['weekday'] = v_train['weekday']
    #volume_test['weekday'] = v_test['weekday']
    
    feature_train = df_train_v.drop('volume', axis = 1)
    feature_test = df_test_v.drop('volume',axis = 1)
    values_train = df_train_v['volume'].values
    values_test = df_test_v['volume'].values
    
    return feature_train, feature_test, values_train, values_test


# In[78]:


feature_train, feature_test, values_train, values_test = feature_format()


# In[81]:


feature_test.count()


# In[82]:


regressor = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=10, oob_score=True)


# In[91]:


regressor.fit(feature_train[['tollgate_id', 'direction', 'hour', 'miniute', 'am_pm']], values_train)


# In[92]:


y_pred = regressor.predict(feature_test[['tollgate_id', 'direction', 'hour', 'miniute', 'am_pm']])


# In[94]:


values_test

