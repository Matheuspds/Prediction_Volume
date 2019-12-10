
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from datetime import time
import matplotlib.pyplot as pplot
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor


# In[4]:


volume_df = pd.read_csv("../dataset/volume(table 6)_test1.csv")


# In[5]:


volume_df['date_time'] = pd.to_datetime(volume_df['time'], format = '%Y-%m-%d %H:%M:%S')

volume_df['t'] = volume_df['date_time'].dt.time


# In[8]:


volume_df['time_window'] = volume_df['t'].apply(get_timewindow)


# In[7]:


#Função que será usada para obter a janela de tempo de 20 minutos
def get_timewindow(t):
        time_window = 20
        if t.minute < time_window:
            window = [time(t.hour, 0), time(t.hour,20)]
        elif t.minute < time_window*2:
            window = [time(t.hour, 20), time(t.hour, 40)]
        else:
            try:
                window = [time(t.hour, 40), time(t.hour + 1, 0)]
            except ValueError:
                window = [time(t.hour, 40), time(0,0,0)]
        s_window = '[' + str(window[0]) + ',' + str(window[1]) + ')'
        return s_window

def get_hour(t):
        return t.hour


# In[12]:


volume_df["week"] = volume_df["date_time"].apply(lambda x: x.dayofweek)


# In[14]:


volume_df['time'] =  pd.to_datetime(volume_df['time'] , format='%Y-%m-%d %H:%M:%S')
#volume_df = volume_df.set_index(['time_window'])

volume_df = volume_df.groupby([pd.Grouper(freq='20T', key="time"), 'tollgate_id', 'direction', 'time_window']).size()       .reset_index().rename(columns = {0:'volume'})


# In[17]:


volume_df["week"] = volume_df["time"].apply(lambda x: x.dayofweek)


# In[19]:


volume_df.head()


# In[20]:


volume_df.to_csv("data_process_final/teste_final_para_weekday.csv", index=False)

