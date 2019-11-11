
# coding: utf-8

# In[24]:


import pandas as pd
from datetime import time


# In[12]:


df_final = pd.read_csv("../dataset/volume(table 6)_test2.csv")
df_final.head()


# In[21]:


df_final = df_final.rename(columns = {'date_time':'time','tollgate':'tollgate_id','is_etc':'has_etc','veh_type':'vehicle_type','model':'vehicle_model'})

df_final.head()


# In[22]:


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


# In[25]:


df_final['t'] = df_final['time'].dt.time

df_final['time_window'] = df_final['t'].apply(get_timewindow)


# In[26]:


df_final['time'] =  pd.to_datetime(df_final['time'] , format='%Y-%m-%d %H:%M:%S')

df_final = df_final.groupby([pd.Grouper(freq='20T', key='time'), 'tollgate_id', 'direction', 'time_window']).size()       .reset_index().rename(columns = {0:'volume'})


# In[27]:


df_final.head()


# In[28]:


df_remove = df_final.loc[(df_final['volume'] == 0)]

ultimo_df = df_final.drop(df_remove.index)
ultimo_df.head()


# In[29]:


ultimo_df.head()


# In[30]:


ultimo_df.to_csv("resultado_real_teste.csv", index=False)

