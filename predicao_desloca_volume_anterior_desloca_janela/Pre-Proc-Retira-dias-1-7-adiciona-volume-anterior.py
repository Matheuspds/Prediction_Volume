
# coding: utf-8

# In[34]:


import pandas as pd
from datetime import time
import numpy as np

path = "data_after_process/"


# In[175]:


df_train1 = pd.read_csv(path+"test_15.csv")


# In[176]:


df_train1.head()


# In[177]:


df_train1['time_start'] = pd.to_datetime(df_train1['time_start'], format = '%Y-%m-%d %H:%M:%S')


# In[178]:


df_train1.head()


# In[179]:


df_train1['weekday'] = df_train1['time_start'].dt.dayofweek + 1


# In[180]:


df_train1['t'] = df_train1['time_start'].dt.time


# In[181]:


df_train1['weekend'] = df_train1['weekday'].apply(lambda x: 0 if x < 6 else 1)


# In[182]:


def get_hour(t):
        return t.hour


# In[183]:


df_train1['hour'] = df_train1['t'].apply(get_hour)


# In[184]:


df_train1['date'] = df_train1['time_start'].dt.date


# In[185]:


del df_train1['t']


# In[186]:


df_train1.head()


# In[187]:


df_train1["volume_anterior"] = df_train1.groupby(['direction', 'tollgate_id'])["volume"].transform("shift")
df_train1["volume_anterior"] = df_train1.groupby(['direction', 'tollgate_id'])["volume_anterior"].fillna(method="ffill").fillna(0)


# In[168]:


df_train1.head()


# In[188]:


#Adicionando valor da janela de tempo (2x) anterior na janela de tempo atual
df_train1["volume_anterior_2"] = df_train1.groupby(['direction', 'tollgate_id'])["volume"].transform("shift", 2)
df_train1["volume_anterior_2"] =df_train1.groupby(['direction', 'tollgate_id'])["volume_anterior_2"].fillna(method="ffill").fillna(0)
df_train1.head()


# In[189]:


df_train1['media_volume'] = df_train1.groupby(['time_window', 'weekday', 'direction', 'tollgate_id'])["volume"].transform(np.mean)


# In[190]:


df_train1.head()


# In[172]:


df_train1['desvio_padrao'] = df_train1.groupby(['time_window', 'weekday', 'direction', 'tollgate_id'])["volume"].transform(np.std)

df_train1['desvio_padrao'].fillna(df_train1.groupby(['time_window', 'weekday', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)


# In[173]:


df_train1.head()


# In[174]:


df_train1.to_csv('data_after_process_valores_anteriores/test10_valor_anterior.csv', index = False)

