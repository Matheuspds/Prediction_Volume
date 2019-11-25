
# coding: utf-8

# In[18]:


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from datetime import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import math


# In[5]:


pd_train = pd.read_csv('../processed_train_volume2.csv')


# In[6]:


pd_train.head()


# In[9]:


pd_train['time'] =  pd.to_datetime(pd_volume_train['time'] , format='%Y-%m-%d %H:%M:%S')

pd_train = pd_train.groupby([pd.Grouper(freq='20T', key='time'), 'tollgate_id', 'direction', 'time_window', 'date', 'hour']).size()       .reset_index().rename(columns = {0:'volume'})


# In[10]:


pd_train['weekday'] = pd_train['time'].dt.dayofweek + 1


# In[23]:


pd_train[(pd_train['date'] == '2016-10-01') & (pd_train['hour'] == 8)].head()


# In[24]:


df_remove = pd_train.loc[(pd_train['date'] >= '2016-10-01') 
                         & (pd_train['date'] <= '2016-10-07') 
                        ]

pd_train = pd_train.drop(df_remove.index)


# In[25]:


pd_train.head()


# In[26]:


pd_train[(pd_train['date'] == '2016-10-01') & (pd_train['hour'] == 8)].head()


# In[29]:


pd_train.tail()


# In[27]:


freq = "20min"


# In[30]:


# movimenta a time_window 0 5 10 15 minutos para os dados de treino
range_1 = pd.date_range("2016-09-19 00:00:00", "2016-10-17 00:00:00", freq=freq)
range_2 = pd.date_range("2016-09-19 00:05:00", "2016-10-17 00:00:00", freq=freq)
range_3 = pd.date_range("2016-09-19 00:10:00", "2016-10-17 00:00:00", freq=freq)
range_4 = pd.date_range("2016-09-19 00:15:00", "2016-10-17 00:00:00", freq=freq)


# In[31]:


def run(df,rng):
    rng_length = len(rng)
    result_dfs = []
    for this_direction in range(2):
        for this_tollgate_id in range(1, 4):
            time_start_list = []
            volume_list = []
            direction_list = []
            tollgate_id_list = []

            this_df = df[(df.tollgate_id == this_tollgate_id) & (df.direction == this_direction)]
            if len(this_df) > 0:
                for ind in range(rng_length - 1):
                    this_df_time_window = this_df[(this_df.time >= rng[ind]) & (this_df.time < rng[ind + 1])]
                    volume_list.append(len(this_df_time_window))

                    time_start_list.append(rng[ind])

                result_df = pd.DataFrame({'time_start': time_start_list,
                                          'volume': volume_list,
                                          'direction': [this_direction] * (rng_length - 1),
                                          'tollgate_id': [this_tollgate_id] * (rng_length - 1),
                }
                )
                result_dfs.append(result_df)

    d = pd.concat(result_dfs)

    if type == 'test':
        d['hour'] = d['time_start'].apply(lambda x: x.hour)
        dd = d[d.hour.isin([6, 7, 15, 16])]
    return d


# In[32]:


df_train_0 = run(pd_train,range_1)


# In[33]:


df_train_0.head()

