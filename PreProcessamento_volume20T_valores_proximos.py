
# coding: utf-8

# In[21]:


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


# In[71]:


v_train = pd.read_csv('dados_treino_volume_com_valor_anterior.csv')
v_test = pd.read_csv('dados_teste_volume_com_valor_anterior.csv')


# In[72]:


df_remove = v_train.loc[(v_train['date'] >= '2016-10-01') 
                         & (v_train['date'] <= '2016-10-07') 
                        ]

v_train = v_train.drop(df_remove.index)


# In[83]:


df_train_list = [v_train,]
random.shuffle(df_train_list)
df_train = pd.concat(df_train_list)

df_train_grouped = df_train.groupby(["tollgate_id", "direction"])
df_test_grouped = v_test.groupby(["tollgate_id", "direction"])
result = []
oob = []
for key, train_data in df_train_grouped:

    test_data = df_test_grouped.get_group(key)
    len_train = len(train_data)
    train_data = train_data.append(test_data)[train_data.columns.tolist()]


    #train_data = feature_transform_knn(key, train_data)

    regressor_cubic = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=10, oob_score=True)

    train_data = pd.DataFrame.reset_index(train_data)
    train_data = train_data.drop("index", axis=1)
    y = train_data.ix[:len_train - 1, :]["volume"]


    x = train_data.ix[:len_train - 1, 5:]
    x1 = train_data.ix[len_train:, 5:]

    regressor_cubic.fit(x, y)
    yhat = regressor_cubic.predict(x1)

    test_data["volume"] = yhat
    result.append(test_data[['tollgate_id', 'time_window', 'direction', 'volume']])


df_result = pd.concat(result, axis=0)


# In[74]:


df_result.head()


# In[75]:


test_0 = pd.read_csv("predicao-com-serie/data_after_process/test_0.csv")


# In[76]:


df_remove = test_0.loc[(test_0['volume'] == 0)]

test_0 = test_0.drop(df_remove.index)
test_0.head()


# In[77]:


volume_real = test_0['volume'].values
volume_prediction = df_result['volume'].values


# In[78]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[79]:


mean_absolute_percentage_error(volume_real, volume_prediction)


# In[80]:


mean_absolute_error(volume_real, volume_prediction)


# In[81]:


volume_real


# In[82]:


volume_prediction

