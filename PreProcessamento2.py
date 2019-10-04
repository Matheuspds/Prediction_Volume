
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from datetime import time
import matplotlib.pyplot as pplot
import math


# In[33]:


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


# In[34]:


volume_df['vehicle_type'] = volume_df['vehicle_model'].apply(lambda x: 0 if x < 5 else 1)


# In[35]:


volume_df['tollgate_id_string'] = volume_df['tollgate_id']
volume_df['tollgate_id_string'] = volume_df['tollgate_id'].replace({1: "1S", 2: "2S", 3: "3S"})

volume_df.head()


# In[38]:


def getTimeFormat(t):
    value = int(math.floor(t.minute / 20) * 20)
    return value

def get_timewindow(t):
        delta = 20
        if t.minute < delta:
            window = [time(t.hour, 0), time(t.hour,20)]
        elif t.minute < delta*2:
            window = [time(t.hour, 20), time(t.hour, 40)]
        else:
            try:
                window = [time(t.hour, 40), time(t.hour + 1, 0)]
            except ValueError:
                window = [time(t.hour, 40), time(0,0,0)]
        s_window = '[' + str(window[0]) + ',' + str(window[1]) + ')'
        return s_window

if hasattr(time, 'strptime'):
    #python 2.6
    strptime = time.strptime
else:
    #python 2.4 equivalent
    strptime = lambda date_string, format: time(*(time.strptime(date_string, format)[0:6]))

volume_df['time'] =  pd.to_datetime(volume_df['time'] , format='%Y-%m-%d %H:%M:%S')
#volume_df = volume_df.set_index(['time'])

volume_df = volume_df.groupby([pd.Grouper(key='time',freq='20Min'), 'tollgate_id', 'direction']).size()       .reset_index().rename(columns = {0:'volume'})


# In[39]:


volume_df.head()

