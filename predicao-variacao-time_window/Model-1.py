
# coding: utf-8

# In[148]:


import pandas as pd
import numpy as np
import random
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
path_final = 'data_final/'


# In[149]:


df_test1 = pd.read_csv(path_final+"test_final_0.csv")
df_test2 = pd.read_csv(path_final+"test_final_5.csv")
df_test3 = pd.read_csv(path_final+"test_final_10.csv")
df_test4 = pd.read_csv(path_final+"test_final_15.csv")
df_train1 = pd.read_csv(path_final+"train_final_0.csv")
df_train2 = pd.read_csv(path_final+"train_final_5.csv")
df_train3 = pd.read_csv(path_final+"train_final_10.csv")
df_train4 = pd.read_csv(path_final+"train_final_15.csv")


# In[150]:


df_train_list = [df_train1,df_train2, df_train3, df_train4]
df_test_list = [df_test1, df_test2, df_test3, df_test4]


# In[151]:


random.shuffle(df_train_list)
random.shuffle(df_test_list)


# In[152]:


df_train = pd.concat(df_train_list)


# In[154]:


df_test = pd.concat(df_test_list)


# In[155]:


df_train.count()


# In[156]:


df_test.head()


# In[157]:


df_remove = df_test.loc[(df_test['volume'] == 0)]

df_test = df_test.drop(df_remove.index)


# In[158]:


def feature_format():
    #pd_volume_train = pd_volume_train.set_index(['time'])
    #pd_volume_test = pd_volume_test.set_index(['time'])
    #volume_train = v_train.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #volume_test = v_test.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #print(volume_train)                
    #x = pd.Series(df_train['time_window'].unique())
    #s = pd.Series(range(len(x)),index = x.values)
    #df_train['window_n'] = df_train['time_window'].map(s)
    #df_test['window_n'] = df_test['time_window'].map(s)
#        print vol_test.tail()
    #volume_train['weekday'] = v_train['weekday']
    #volume_test['weekday'] = v_test['weekday']
    
    feature_train = df_train.drop('volume', axis = 1)
    feature_test = df_test.drop('volume',axis = 1)
    values_train = df_train['volume'].values
    values_test = df_test['volume'].values
    
    return feature_train, feature_test, values_train, values_test


# In[159]:


feature_train, feature_test, values_train, values_test = feature_format()
feature_train.head()


# In[160]:


def feature_transform_split(data):
    data["precipitation"] = data[["precipitation"]].fillna(value=0)
    data["rel_humidity"] = data[["rel_humidity"]].fillna(value=50)

    data["precipitation"] = data["precipitation"].apply(lambda x: x > 0)
    data["rel_humidity"] = data["rel_humidity"].apply(lambda x: x > 90)

    data = data.drop("precipitation", axis=1)
    return data


# In[138]:


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

    data = pd.concat([data, pd.get_dummies(data['holiday'])], axis=1)
    data = data.drop("holiday", axis=1)
    #
    # data = pd.concat([data, pd.get_dummies(data['first_last_workday'])], axis=1)
    #data = data.drop("first_last_workday", axis=1)

    #data = data.drop("day_of_week", axis=1)

    if (key == 1):
        data = pd.concat([data, pd.get_dummies(data['tollgate_id'])], axis=1)
        # data["tollgate_id1"] = data['tollgate_id']
        data["direction1"] = data['direction']
    return data


# In[162]:


feature_train = feature_transform_split(feature_train)


# In[161]:


feature_test = feature_transform_split(feature_test)


# In[141]:


feature_train.head()


# In[142]:


regressor_cubic = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=10, oob_score=True)


# In[143]:


#x = pd.Series(df_train['time_window'].unique())
#s = pd.Series(range(len(x)),index = x.values)
#df_train['window_n'] = df_train['time_window'].map(s)
#df_test['window_n'] = df_test['time_window'].map(s)
df_train_grouped = df_train.groupby(["tollgate_id", "direction"])
df_test_grouped = df_test.groupby(["tollgate_id", "direction"])
result = []
oob = []
for key, train_data in df_train_grouped:

    test_data = df_test_grouped.get_group(key)
    len_train = len(train_data)
    train_data = train_data.append(test_data)[train_data.columns.tolist()]
    train_data = feature_transform_split_complete(key, train_data)

    regressor_cubic = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=10, oob_score=True)
    train_data = pd.DataFrame.reset_index(train_data)
    train_data = train_data.drop("index", axis=1)
    y = train_data.ix[:len_train - 1, :]["volume"]
    x = train_data.ix[:len_train - 1, 8:]
    x1 = train_data.ix[len_train:, 8:]
    regressor_cubic.fit(x, y)
    yhat = regressor_cubic.predict(x1)


# In[147]:


len(yhat)
len(values_test)


# In[85]:


regressor_cubic.fit(feature_train[['tollgate_id', 'direction', 'weekday', 'weekend', 'hour', 'minute', 'holiday', 'am_pm', 'rel_humidity']], values_train)


# In[86]:


y_pred = regressor_cubic.predict(feature_test[['tollgate_id', 'direction', 'weekday', 'weekend', 'hour', 'minute', 'holiday', 'am_pm', 'rel_humidity']])


# In[178]:


df_value_pred = pd.read_csv('data_final/resultado_final.csv')
df_value_real = pd.read_csv('../dataset/test2_20min_avg_volume_test.csv')
values_pred = df_value_pred['volume'].values
values_reais = df_value_real['volume'].values
len(values_reais)


# In[89]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[179]:


mean_absolute_percentage_error(values_reais, values_pred)


# In[116]:


mape


# In[93]:


len(y_pred)


# In[94]:


len(values_test)


# In[180]:


mape = np.mean(np.abs((values_pred - values_reais)/values_reais))


# In[181]:


mape


# In[182]:


values_pred


# In[184]:


values_reais

