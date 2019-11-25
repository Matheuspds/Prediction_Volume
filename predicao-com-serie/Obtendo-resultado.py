
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import random


# In[2]:


df_test = pd.read_csv("test2.csv")

df_train0 = pd.read_csv("train.csv")
df_train1 = pd.read_csv("train1.csv")
df_train2 = pd.read_csv("train2.csv")
df_train3 = pd.read_csv("train3.csv")
df_train_list = [df_train0,df_train1, df_train2, df_train3]
df_train3.head()


# In[3]:


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


# In[4]:


#df_hue = pd.concat(df_train_list)
#df_hue.to_csv("result/result_split_rf_huehue"+"huehue"+".csv", index=False)
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
result = []
oob = []
for key, train_data in df_train_grouped:
    test_data = df_test_grouped.get_group(key)
    len_train = len(train_data)
    train_data = train_data.append(test_data)[train_data.columns.tolist()]
    train_data = feature_transform_split(key, train_data)
    train_data.head()

    regressor_cubic = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=10, oob_score=True)

    train_data = pd.DataFrame.reset_index(train_data)
    train_data = train_data.drop("index", axis=1)
    y = train_data.ix[:len_train - 1, :]["volume"]

    
    x = train_data.ix[:len_train - 1, 8:]
    x1 = train_data.ix[len_train:, 8:]
    regressor_cubic.fit(x, y)
    print(x1)
    resultado_obtido = regressor_cubic.predict(x1)

    #df_h = test_data
    #df_h.to_csv("result/result_split_rf_huehue"+"teste"+".csv", index=False)

    test_data["volume"] = resultado_obtido
    result.append(test_data[['tollgate_id', 'time_window', 'direction', 'volume']])


# In[8]:


df_result = pd.concat(result, axis=0)
df_result.to_csv("resultado_final"+"verificar"+".csv", index=False)
df_result.head()


# In[117]:


df_ = pd.read_csv("test2.csv")
df_.count()


# In[92]:


df_test_real = pd.read_csv("resultado_real_teste.csv")

#df_train0 = pd.read_csv("train.csv")
#df_train1 = pd.read_csv("train1.csv")


# In[93]:


df_test_real.head()


# In[94]:


df_result.head()


# In[ ]:


def feature_format():
    v_train = pd.read_csv('dados_treino_volume_com_valor_anterior.csv')
    v_test = pd.read_csv('dados_teste_volume_com_valor_anterior.csv')
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


# In[95]:


values_test_real = df_test_real['volume'].values
values = df_result['volume'].values.astype(int)
values_test_prediction = values


# In[97]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[98]:


mean_absolute_percentage_error(values_test_real, values_test_prediction)


# In[96]:


values_test_prediction


# In[84]:


values_test_real


# In[100]:


mape = np.mean(np.abs((values_test_prediction - values_test_real)/values_test_real))
mape

