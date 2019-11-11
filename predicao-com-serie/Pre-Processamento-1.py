
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

path = "data_after_process/"


# In[5]:


#Buscando criar novas colunas com dados de data, hora, minuto, time para a determinada linha do dataset
def df_filter(df_volume):
    df_volume["time"] = df_volume["time_start"]
    df_volume["date"] = df_volume["time"].apply(lambda x: pd.to_datetime(x[: 10]))
    df_volume["hour"] = df_volume["time"].apply(lambda x: int(x[11: 13]))
    df_volume["miniute"] = df_volume["time"].apply(lambda x: int(x[14: 16]))
    df_volume["time_window"] = df_volume["time"]
    df_volume = df_volume[["tollgate_id", "time_window", "direction", "volume", "time", "date", "hour", "miniute"]]
    return  df_volume


# In[3]:


df_test = pd.read_csv(path+"test_0.csv" )
df_train1 = pd.read_csv(path+"train_0.csv")
df_train2 = pd.read_csv(path+"train_5.csv")
df_train3 = pd.read_csv(path+"train_10.csv")
df_train4 = pd.read_csv(path+"train_15.csv")

df_filter(df_test).to_csv(path+"test_filter_0.csv",index=False)
df_filter(df_train1).to_csv(path+"train_filter_0.csv",index=False)
df_filter(df_train2).to_csv(path+"train_filter_5.csv",index=False)
df_filter(df_train3).to_csv(path+"train_filter_10.csv",index=False)
df_filter(df_train4).to_csv(path+"train_filter_15.csv",index=False)


# In[8]:


df_hue = pd.read_csv(path+"train_filter_5.csv")


# In[9]:


df_hue.head()

