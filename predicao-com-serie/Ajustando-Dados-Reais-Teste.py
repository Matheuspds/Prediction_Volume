
# coding: utf-8

# In[52]:


import pandas as pd
import numpy as np
from datetime import time
path = "../dataset/"
freq = "20min"


# In[54]:


df_test = pd.read_csv("resultado_real_teste.csv", parse_dates=['time'])
df_test = df_test.rename(columns = {'date_time':'time','tollgate':'tollgate_id','is_etc':'has_etc','veh_type':'vehicle_type','model':'vehicle_model'})

df_test.head()


# In[55]:


del df_test['time_window']


# In[57]:


df_test['time_window'] = df_test['time']


# In[59]:


del df_test['time']


# In[63]:


df_test.head()
df_test.to_csv('result/resultado_real.csv', index=False)


# In[49]:


def df_filter(df_volume):
    df_volume["time"] = df_volume["time_start"]
    df_volume["time_window"] = df_volume["time"]
    df_volume = df_volume[["tollgate_id", "time_window", "direction", "volume"]]
    return  df_volume


# In[50]:


df_test_final = df_filter(df_test)


# In[51]:


df_test_final

