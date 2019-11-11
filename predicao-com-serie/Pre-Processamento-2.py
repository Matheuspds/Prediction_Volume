
# coding: utf-8

# In[10]:


import pandas as pd
import os


# In[17]:


#Ajustando dados de clima para quem sabe poderem ser usados também

path = "../dataset/"
df_weather = pd.read_csv(path+"weather (table 7)_training_update.csv")
df_weather = df_weather.append(
pd.read_csv(path+"weather (table 7)_test1.csv")).append(pd.read_csv(path+"weather (table 7)_2.csv"))

df_weather.tail()


# In[15]:


df_weather["hour"] = df_weather["hour"]
df_weather[
    ["date", "hour", "pressure", "sea_pressure", "wind_direction", "wind_speed", "temperature", "rel_humidity",
        "precipitation"]].to_csv("data_after_process/feature_clima.csv", index=False)


# In[16]:


df_weather.tail()

