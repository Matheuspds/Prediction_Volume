
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import time
path = "data_after_process/"
path_result ="dados_sem_feriado/"


# In[2]:


df_test1 = pd.read_csv(path+"test_0.csv" )
df_test2 = pd.read_csv(path+"test_5.csv" )
df_test3 = pd.read_csv(path+"test_10.csv" )
df_test4 = pd.read_csv(path+"test_15.csv" )
df_train1 = pd.read_csv(path+"train_0.csv")
df_train2 = pd.read_csv(path+"train_5.csv")
df_train3 = pd.read_csv(path+"train_10.csv")
df_train4 = pd.read_csv(path+"train_15.csv")


# In[3]:


def get_minute(t):
        return t.minute


# In[4]:


def get_hour(t):
        return t.hour


# In[20]:


def get_am_pm(t):
    if(t < 12):
        return 1
    else: 
        return 0


# In[28]:


def get_column_date(df_train):
    df_train['time_start'] = pd.to_datetime(df_train['time_start'], format = '%Y-%m-%d %H:%M:%S')
    df_train['date'] = df_train['time_start'].dt.date
    
    df_train['t'] = df_train['time_start'].dt.time
    
    df_train['weekday'] = df_train['time_start'].dt.dayofweek + 1

    #Adicionando valores para saber se é referente a um fim de semana ou não
    df_train['weekend'] = df_train['weekday'].apply(lambda x: 0 if x < 6 else 1)
    
    df_train['hour'] = df_train['t'].apply(get_hour)
    
    df_train['day'] = df_train['time_start'].dt.day
    
    df_train['am_pm'] = df_train['hour'].apply(get_am_pm)
    
    del df_train['t']
    
    return df_train


# In[29]:


df_train1 = get_column_date(df_train1)

df_train2 = get_column_date(df_train2)

df_train3 = get_column_date(df_train3)

df_train4 = get_column_date(df_train4)


# In[45]:


df_train2[(df_train2['day'] == 19) & (df_train2['hour'] == 8)].head()


# In[52]:


df_test1 = get_column_date(df_test1)

df_test2 = get_column_date(df_test2)

df_test3 = get_column_date(df_test3)

df_test4 = get_column_date(df_test4)


# In[53]:


df_test1.head()


# In[46]:


def remove_feriado(df_train):
    df_remove = df_train.loc[(df_train['day'] >= 1) & (df_train['day'] <= 7) ]
    df_train = df_train.drop(df_remove.index)
    return df_train


# In[47]:


df_train1 = remove_feriado(df_train1)

df_train2 = remove_feriado(df_train2)

df_train3 = remove_feriado(df_train3)

df_train4 = remove_feriado(df_train4)


# In[50]:


df_train2[(df_train2['day'] == 7) & (df_train2['hour'] == 8)].head()


# In[51]:


df_train1.to_csv(path_result+"train1.csv", index=False)
df_train2.to_csv(path_result+"train2.csv", index=False)
df_train3.to_csv(path_result+"train3.csv", index=False)
df_train4.to_csv(path_result+"train4.csv", index=False)


# In[54]:


df_test1.to_csv(path_result+"test1.csv", index=False)
df_test2.to_csv(path_result+"test2.csv", index=False)
df_test3.to_csv(path_result+"test3.csv", index=False)
df_test4.to_csv(path_result+"test4.csv", index=False)

