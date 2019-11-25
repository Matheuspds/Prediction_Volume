
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import time
path = "data_after_process/"


# In[8]:


def get_hour(t):
        return t.hour


# In[10]:


def get_minute(t):
        return t.minute


# In[13]:


#Buscando criar novas colunas com dados de data, hora, minuto, time para a determinada linha do dataset
def df_filter(volume_df):
    #Ajustando o formato da coluna time
    volume_df['time_start'] = pd.to_datetime(volume_df['time_start'], format = '%Y-%m-%d %H:%M:%S')
    #Representam os feriados
    holiday1 = [pd.Timestamp('2016-10-1'), pd.Timestamp('2016-10-8')]
    holiday2 = [pd.Timestamp('2016-9-15'), pd.Timestamp('2016-9-18')]

    #Adiciona valores para os dias da semana
    volume_df['weekday'] = volume_df['time_start'].dt.dayofweek + 1

    #Classificar cada atributo de time aplicando a janela de tempo de vinte minutos
    volume_df['t'] = volume_df['time_start'].dt.time

    #Adicionando valores para saber se é referente a um fim de semana ou não
    volume_df['weekend'] = volume_df['weekday'].apply(lambda x: 0 if x < 6 else 1)

    volume_df['date'] = volume_df['time_start'].dt.date
    
    volume_df['hour'] = volume_df['t'].apply(get_hour)

    volume_df['minute'] = volume_df['t'].apply(get_minute)

    volume_df['holiday'] = volume_df['time_start'].between(holiday1[0],holiday1[1])                            | volume_df['time_start'].between(holiday2[0],holiday2[1])
    del volume_df['t']
    
    return volume_df


# In[14]:


df_test1 = pd.read_csv(path+"test_0.csv" )
df_test2 = pd.read_csv(path+"test_5.csv" )
df_test3 = pd.read_csv(path+"test_10.csv" )
df_test4 = pd.read_csv(path+"test_15.csv" )
df_train1 = pd.read_csv(path+"train_0.csv")
df_train2 = pd.read_csv(path+"train_5.csv")
df_train3 = pd.read_csv(path+"train_10.csv")
df_train4 = pd.read_csv(path+"train_15.csv")

df_filter(df_test1).to_csv(path+"test_filter_0.csv",index=False)
df_filter(df_test2).to_csv(path+"test_filter_5.csv",index=False)
df_filter(df_test3).to_csv(path+"test_filter_10.csv",index=False)
df_filter(df_test4).to_csv(path+"test_filter_15.csv",index=False)
df_filter(df_train1).to_csv(path+"train_filter_0.csv",index=False)
df_filter(df_train2).to_csv(path+"train_filter_5.csv",index=False)
df_filter(df_train3).to_csv(path+"train_filter_10.csv",index=False)
df_filter(df_train4).to_csv(path+"train_filter_15.csv",index=False)

