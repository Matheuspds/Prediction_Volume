
# coding: utf-8

# In[1]:


import pandas as pd
import os

path = "../dataset/"
freq = "20min"


# In[2]:


df_train1 = pd.read_csv(path+"volume(table 6)_training.csv", parse_dates=['time'])
df_train2 = pd.read_csv(path+"volume(table 6)_training2.csv", parse_dates=['date_time'])
df_train2 = df_train2.rename(columns = {'date_time':'time','tollgate':'tollgate_id','is_etc':'has_etc','veh_type':'vehicle_type','model':'vehicle_model'})
df_train = df_train1.append(df_train2)
df_train.tail()


# In[8]:


df_test = pd.read_csv(path+"volume(table 6)_test2.csv", parse_dates=['date_time'])
df_test = df_test.rename(columns = {'date_time':'time','tollgate':'tollgate_id','is_etc':'has_etc','veh_type':'vehicle_type','model':'vehicle_model'})
df_test.head()


# In[9]:


# movimenta a time_window 0 5 10 15 minutos para os dados de treino
range_1 = pd.date_range("2016-09-19 00:00:00", "2016-10-25 00:00:00", freq=freq)
range_2 = pd.date_range("2016-09-19 00:05:00", "2016-10-25 00:00:00", freq=freq)
range_3 = pd.date_range("2016-09-19 00:10:00", "2016-10-25 00:00:00", freq=freq)
range_4 = pd.date_range("2016-09-19 00:15:00", "2016-10-25 00:00:00", freq=freq)


# In[10]:


# movimenta a time_window 0 5 10 15 minutos para os dados de teste
range_5 = pd.date_range("2016-10-25 00:00:00", "2016-11-01 00:00:00", freq=freq)


# In[11]:


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


# In[13]:


run(df_train,range_1).to_csv("data_after_process/train_0.csv",index= False)
run(df_train,range_2).to_csv("data_after_process/train_5.csv",index= False)
run(df_train,range_3).to_csv("data_after_process/train_10.csv",index= False)
run(df_train,range_4).to_csv("data_after_process/train_15.csv",index= False)
run(df_test,range_5).to_csv("data_after_process/test_0.csv",index= False)


# In[19]:


df_hue = pd.read_csv("data_after_process/train_10.csv")


# In[20]:


df_hue.count()


# In[23]:


df_hue2 = pd.read_csv("data_after_process/test_0.csv")
df_hue2.count()

