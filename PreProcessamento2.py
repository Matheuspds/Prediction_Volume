
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from datetime import datetime
import matplotlib.pyplot as pplot
import math


# In[30]:


volume_df = pd.read_csv("dataset/volume(table 6)_training.csv")
volume_df.head()


# In[5]:


# Descrição das features:
# time           datetime  Data e Hora em que o veículo passa pelo pedágio;
# tollgate_id    string    Identificador do pedágio;
# direction      string    0: entra na rodovia pelo pedágio; 1: sai da rodovia pelo pedágio;
# vehicle_model  int       Um número randomico que indica a capacidade do veículo;
# has_etc        string    Indica se o veículo possui ou não o sistema ETC; 0 - NÃO, 1 - SIM
# vehicle_type   string    0: veículo de passageiro; 1: veículo de carga
# weekday        int       Representa os dias da semana
# weekend        int       1: Para quando for fim de semana; 0: Para quando não for fim de semana


# In[6]:


#Retirando os valores nulos da coluna vehicle_type pelo modelo do veículo.
    #No vehicle_type indica 0 para veículo de passageiros e 1 para carga.
    #Poderíamos verificar a partir do modelo do veiculo, para veiculo com capacidade de até 4
    #Ficou definido que sera para passageiro, sendo maior que 4 será veiculo de carga


# In[31]:


volume_df['vehicle_type'] = volume_df['vehicle_model'].apply(lambda x: 0 if x < 5 else 1)


# In[32]:


volume_df['tollgate_id_string'] = volume_df['tollgate_id']
volume_df['tollgate_id_string'] = volume_df['tollgate_id'].replace({1: "1S", 2: "2S", 3: "3S"})

volume_df.head()


# In[36]:


def getTimeFormat(wd):
    return '[{},{})'.format(str(wd), str(wd+timedelta(minutes=20)))

def time_to_window(x):
    dt = datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
    dtmin = int(dt.minute / 20) * 20
    dtwindow = datetime(dt.year, dt.month, dt.day, dt.hour, dtmin, 0)
    return dtwindow

if hasattr(time, 'strptime'):
    #python 2.6
    strptime = time.strptime
else:
    #python 2.4 equivalent
    strptime = lambda date_string, format: time(*(time.strptime(date_string, format)[0:6]))

#volume_df['time'] =  pd.to_datetime(volume_df['time'] , format='%Y-%m-%d %H:%M:%S')
volume_df['window'] = volume_df['time'].apply(time_to_window)
#volume_df = volume_df.groupby(['tollgate_id','direction','window']).count()['time']


volume_df = volume_df.groupby([pd.Grouper(key='window'), 'tollgate_id', 'direction']).size()      .reset_index().rename(columns = {0:'volume'})


# In[37]:


volume_df

