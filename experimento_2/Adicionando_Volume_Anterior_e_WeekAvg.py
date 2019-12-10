
# coding: utf-8

# In[25]:


import pandas as pd
from datetime import time
import numpy as np

path = "dados_sem_feriado/"
path_result="dados_finais/"


# In[2]:


df_train1 = pd.read_csv(path+"train1.csv")
df_train2 = pd.read_csv(path+"train2.csv")
df_train3 = pd.read_csv(path+"train3.csv")
df_train4 = pd.read_csv(path+"train4.csv")
df_test1 = pd.read_csv(path+"test1.csv")
df_test2 = pd.read_csv(path+"test2.csv")
df_test3 = pd.read_csv(path+"test3.csv")
df_test4 = pd.read_csv(path+"test4.csv")


# In[6]:


def adiciona_volume_anterior(df):
    df["volume_anterior"] = df.groupby(['direction', 'tollgate_id'])["volume"].transform("shift")
    df["volume_anterior"] = df.groupby(['direction', 'tollgate_id'])["volume_anterior"].fillna(method="ffill").fillna(0)
    return df


# In[7]:


def adiciona_volume_anterior_2(df):
    df["volume_anterior_2"] = df.groupby(['direction', 'tollgate_id'])["volume"].transform("shift", 2)
    df["volume_anterior_2"] =df.groupby(['direction', 'tollgate_id'])["volume_anterior_2"].fillna(method="ffill").fillna(0)
    return df


# In[13]:


def adiciona_media_desvio_semana(df):
    df['media_volume'] = df.groupby(['time_window', 'weekday', 'direction', 'tollgate_id'])["volume"].transform(np.mean)
    df['desvio_padrao'] = df.groupby(['time_window', 'weekday', 'direction', 'tollgate_id'])["volume"].transform(np.std)
    df['desvio_padrao'].fillna(df.groupby(['time_window', 'weekday', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
    return df


# In[14]:


df_train1 = adiciona_volume_anterior(df_train1)
df_train2 = adiciona_volume_anterior(df_train2)
df_train3 = adiciona_volume_anterior(df_train3)
df_train4 = adiciona_volume_anterior(df_train4)
df_test1 = adiciona_volume_anterior(df_test1)
df_test2 = adiciona_volume_anterior(df_test2)
df_test3 = adiciona_volume_anterior(df_test3)
df_test4 = adiciona_volume_anterior(df_test4)


# In[17]:


df_train1 = adiciona_volume_anterior_2(df_train1)
df_train2 = adiciona_volume_anterior_2(df_train2)
df_train3 = adiciona_volume_anterior_2(df_train3)
df_train4 = adiciona_volume_anterior_2(df_train4)
df_test1 = adiciona_volume_anterior_2(df_test1)
df_test2 = adiciona_volume_anterior_2(df_test2)
df_test3 = adiciona_volume_anterior_2(df_test3)
df_test4 = adiciona_volume_anterior_2(df_test4)


# In[20]:


df_train1 = adiciona_media_desvio_semana(df_train1)
df_train2 = adiciona_media_desvio_semana(df_train2)
df_train3 = adiciona_media_desvio_semana(df_train3)
df_train4 = adiciona_media_desvio_semana(df_train4)


# In[22]:


def adiciona_media_desvio_semana_test(df):
    df['media_volume'] = df.groupby(['time_window', 'direction', 'tollgate_id'])["volume"].transform(np.mean)
    df['desvio_padrao'] = df.groupby(['time_window', 'direction', 'tollgate_id'])["volume"].transform(np.std)
    df['desvio_padrao'].fillna(df.groupby(['time_window', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
    return df


# In[23]:


df_test1 = adiciona_media_desvio_semana_test(df_test1)
df_test2 = adiciona_media_desvio_semana_test(df_test2)
df_test3 = adiciona_media_desvio_semana_test(df_test3)
df_test4 = adiciona_media_desvio_semana_test(df_test4)


# In[26]:


df_train1.to_csv(path_result+"train1.csv", index=False)
df_train2.to_csv(path_result+"train2.csv", index=False)
df_train3.to_csv(path_result+"train3.csv", index=False)
df_train4.to_csv(path_result+"train4.csv", index=False)


# In[27]:


df_test1.to_csv(path_result+"test1.csv", index=False)
df_test2.to_csv(path_result+"test2.csv", index=False)
df_test3.to_csv(path_result+"test3.csv", index=False)
df_test4.to_csv(path_result+"test4.csv", index=False)

