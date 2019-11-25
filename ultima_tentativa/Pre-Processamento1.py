
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


# In[145]:


volume_df_1 = pd.read_csv("../dataset/volume(table 6)_training.csv")
volume_df_2 = pd.read_csv("../dataset/volume(table 6)_training2.csv")
volume_df_2 = volume_df_2.rename(columns = {'date_time':'time','tollgate':'tollgate_id','is_etc':'has_etc','veh_type':'vehicle_type','model':'vehicle_model'})


# In[146]:


volume_df_list = [volume_df_1, volume_df_2]


# In[147]:


volume_df = pd.concat(volume_df_list)


# In[148]:


volume_df.head()


# In[131]:


volume_df = pd.read_csv("../dataset/volume(table 6)_test2.csv")
volume_df = volume_df.rename(columns = {'date_time':'time','tollgate':'tollgate_id','is_etc':'has_etc','veh_type':'vehicle_type','model':'vehicle_model'})


# In[149]:


volume_df['date_time'] = pd.to_datetime(volume_df['time'], format = '%Y-%m-%d %H:%M:%S')

volume_df['t'] = volume_df['date_time'].dt.time


# In[150]:


volume_df['time_window'] = volume_df['t'].apply(get_timewindow)


# In[151]:


volume_df.head()


# In[152]:


volume_df['time'] =  pd.to_datetime(volume_df['time'] , format='%Y-%m-%d %H:%M:%S')
#volume_df = volume_df.set_index(['time_window'])

volume_df = volume_df.groupby([pd.Grouper(freq='20T', key="time"), 'tollgate_id', 'direction', 'time_window']).size()       .reset_index().rename(columns = {0:'volume'})


# In[153]:


volume_df[(volume_df['volume'] >= 100)].count()


# In[154]:


volume_df["day"] = volume_df["time"].apply(lambda x: x.day)
volume_df["hour"] = volume_df["time"].apply(lambda x: x.hour)
volume_df["minute"] = volume_df["time"].apply(lambda x: x.minute)
volume_df["week"] = volume_df["time"].apply(lambda x: x.dayofweek)
volume_df["weekend"] = volume_df["week"].apply(lambda x: 1 if x >= 5 else 0)


# In[155]:


volume_df[(volume_df['day'] == 25)].head()


# In[156]:


volume_df['am_pm'] = volume_df["hour"].apply(lambda x: ampm(x))


# In[157]:


volume_df[(volume_df['day'] == 26)].head()


# In[158]:


volume_df.to_csv("treino_agregado.csv", index=False)


# In[119]:


def ampm(x):
    if (x <= 12 ):
        return 1
    return 0


# In[6]:


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

