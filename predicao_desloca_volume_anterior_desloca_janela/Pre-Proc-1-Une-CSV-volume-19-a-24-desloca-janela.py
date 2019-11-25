
# coding: utf-8

# In[1]:


import pandas as pd
from datetime import time

path = "../dataset/"
freq = "20min"


# In[3]:


df_train1 = pd.read_csv(path+"volume(table 6)_training.csv", parse_dates=['time'])
df_train2 = pd.read_csv(path+"volume(table 6)_training2.csv", parse_dates=['date_time'])
df_train2 = df_train2.rename(columns = {'date_time':'time','tollgate':'tollgate_id','is_etc':'has_etc','veh_type':'vehicle_type','model':'vehicle_model'})
df_train = df_train1.append(df_train2)
df_train.tail()


# In[4]:


df_test = pd.read_csv(path+"volume(table 6)_test2.csv", parse_dates=['date_time'])
df_test = df_test.rename(columns = {'date_time':'time','tollgate':'tollgate_id','is_etc':'has_etc','veh_type':'vehicle_type','model':'vehicle_model'})
df_test.head()


# In[5]:


# movimenta a time_window 0 5 10 15 minutos para os dados de treino
range_1 = pd.date_range("2016-09-19 00:00:00", "2016-10-25 00:00:00", freq=freq)
range_2 = pd.date_range("2016-09-19 00:05:00", "2016-10-25 00:00:00", freq=freq)
range_3 = pd.date_range("2016-09-19 00:10:00", "2016-10-25 00:00:00", freq=freq)
range_4 = pd.date_range("2016-09-19 00:15:00", "2016-10-25 00:00:00", freq=freq)


# In[6]:


# movimenta a time_window 0 5 10 15 minutos para os dados de teste
range_5 = pd.date_range("2016-10-25 00:00:00", "2016-11-01 00:00:00", freq=freq)
range_6 = pd.date_range("2016-10-25 00:05:00", "2016-11-01 00:00:00", freq=freq)
range_7 = pd.date_range("2016-10-25 00:10:00", "2016-11-01 00:00:00", freq=freq)
range_8 = pd.date_range("2016-10-25 00:15:00", "2016-11-01 00:00:00", freq=freq)


# In[7]:


def run(df,rng):
    rng_length = len(rng)
    result_dfs = []
    for this_direction in range(2):
        for this_tollgate_id in range(1, 4):
            time_start_list = []
            volume_list = []
            direction_list = []
            tollgate_id_list = []

            this_df = df[(df.tollgate_id == this_tollgate_id) & (df.direction == this_direction)]
            if len(this_df) > 0:
                for ind in range(rng_length - 1):
                    this_df_time_window = this_df[(this_df.time >= rng[ind]) & (this_df.time < rng[ind + 1])]
                    volume_list.append(len(this_df_time_window))

                    time_start_list.append(rng[ind])

                result_df = pd.DataFrame({'time_start': time_start_list,
                                          'volume': volume_list,
                                          'direction': [this_direction] * (rng_length - 1),
                                          'tollgate_id': [this_tollgate_id] * (rng_length - 1),
                }
                )
                result_dfs.append(result_df)

    d = pd.concat(result_dfs)

    if type == 'test':
        d['hour'] = d['time_start'].apply(lambda x: x.hour)
        dd = d[d.hour.isin([6, 7, 15, 16])]
    return d


# In[8]:


df_train_0 = run(df_train,range_1)
df_train_5 = run(df_train,range_2)
df_train_10 = run(df_train,range_3)
df_train_15 = run(df_train,range_4)
df_test_0 = run(df_test,range_5)
df_test_5 = run(df_test,range_6)
df_test_10 = run(df_test,range_7)
df_test_15 = run(df_test,range_8)


# In[11]:


df_train_0.head()


# In[12]:


#Função que será usada para obter a janela de tempo de 20 minutos iniciando as 00:00h
def get_timewindow0(t):
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


# In[13]:


#Função que será usada para obter a janela de tempo de 20 minutos iniciando as 00:05h
def get_timewindow5(t):
        time_window = 20
        if t.minute < time_window:
            window = [time(t.hour, 5), time(t.hour,25)]
        elif t.minute < time_window*2:
            window = [time(t.hour, 25), time(t.hour, 45)]
        else:
            try:
                window = [time(t.hour, 45), time(t.hour + 1, 5)]
            except ValueError:
                window = [time(t.hour, 45), time(0,0,0)]
        s_window = '[' + str(window[0]) + ',' + str(window[1]) + ')'
        return s_window

def get_hour(t):
        return t.hour


# In[14]:


#Função que será usada para obter a janela de tempo de 20 minutos iniciando as 00:10h
def get_timewindow10(t):
        time_window = 20
        if t.minute < time_window:
            window = [time(t.hour, 10), time(t.hour,30)]
        elif t.minute < time_window*2:
            window = [time(t.hour, 30), time(t.hour, 50)]
        else:
            try:
                window = [time(t.hour, 50), time(t.hour + 1, 10)]
            except ValueError:
                window = [time(t.hour, 50), time(0,0,0)]
        s_window = '[' + str(window[0]) + ',' + str(window[1]) + ')'
        return s_window

def get_hour(t):
        return t.hour


# In[15]:


#Função que será usada para obter a janela de tempo de 20 minutos iniciando as 00:15h
def get_timewindow15(t):
        time_window = 20
        if t.minute < time_window:
            window = [time(t.hour, 15), time(t.hour,35)]
        elif t.minute < time_window*2:
            window = [time(t.hour, 35), time(t.hour, 55)]
        else:
            try:
                window = [time(t.hour, 55), time(t.hour + 1, 15)]
            except ValueError:
                window = [time(t.hour, 55), time(0,0,0)]
        s_window = '[' + str(window[0]) + ',' + str(window[1]) + ')'
        return s_window

def get_hour(t):
        return t.hour


# In[16]:


def cria_time_window0(df):
    df['time_start'] = pd.to_datetime(df['time_start'], format = '%Y-%m-%d %H:%M:%S')
    df['t'] = df['time_start'].dt.time
    df['time_window'] = df['t'].apply(get_timewindow0)
    del df['t']
    return df


# In[17]:


def cria_time_window5(df):
    df['time_start'] = pd.to_datetime(df['time_start'], format = '%Y-%m-%d %H:%M:%S')
    df['t'] = df['time_start'].dt.time
    df['time_window'] = df['t'].apply(get_timewindow5)
    del df['t']
    return df


# In[18]:


def cria_time_window10(df):
    df['time_start'] = pd.to_datetime(df['time_start'], format = '%Y-%m-%d %H:%M:%S')
    df['t'] = df['time_start'].dt.time
    df['time_window'] = df['t'].apply(get_timewindow10)
    del df['t']
    return df


# In[19]:


def cria_time_window15(df):
    df['time_start'] = pd.to_datetime(df['time_start'], format = '%Y-%m-%d %H:%M:%S')
    df['t'] = df['time_start'].dt.time
    df['time_window'] = df['t'].apply(get_timewindow15)
    del df['t']
    return df


# In[20]:


cria_time_window0(df_train_0).to_csv("data_after_process/train_0.csv",index= False)
cria_time_window5(df_train_5).to_csv("data_after_process/train_5.csv",index= False)
cria_time_window10(df_train_10).to_csv("data_after_process/train_10.csv",index= False)
cria_time_window15(df_train_15).to_csv("data_after_process/train_15.csv",index= False)
cria_time_window0(df_test_0).to_csv("data_after_process/test_0.csv",index= False)
cria_time_window5(df_test_5).to_csv("data_after_process/test_5.csv",index= False)
cria_time_window10(df_test_10).to_csv("data_after_process/test_10.csv",index= False)
cria_time_window15(df_test_15).to_csv("data_after_process/test_15.csv",index= False)

