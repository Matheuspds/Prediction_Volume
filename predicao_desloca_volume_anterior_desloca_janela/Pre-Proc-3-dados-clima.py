
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


path = "../dataset/"
df_weather = pd.read_csv(path+"weather (table 7)_training_update.csv")
df_weather = df_weather.append(pd.read_csv(path+"weather (table 7)_test1.csv")).append(
                pd.read_csv(path+"weather (table 7)_2.csv"))
df_weather.head()


# In[3]:


df_weather[
        ["date", "hour", "pressure", "sea_pressure", "wind_direction", "wind_speed", "temperature", "rel_humidity",
         "precipitation"]].to_csv("data_after_process/feature_clima.csv", index=False)

