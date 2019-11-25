
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np
from datetime import time
from sklearn.metrics import mean_absolute_error


# In[22]:


df_final_real = pd.read_csv("result/resultado_real.csv")
df_final_real.head()


# In[31]:


df_final_real = df_final_real.sort_values(['tollgate_id', 'direction'])
df_final_real.head()


# In[32]:


df_final_predict = pd.read_csv('result/result_split_rf_TESTAR_AGORA.csv')
df_final_predict.head()


# In[33]:


volume_real = df_final_real['volume'].values
volume_real


# In[34]:


volume_predict = df_final_predict['volume'].values
volume_predict


# In[35]:


#Função que calcula o MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[36]:


mean_absolute_percentage_error(volume_real, volume_predict)


# In[37]:


mape = np.mean(np.abs((volume_predict - volume_real)/volume_real))


# In[38]:


mape


# In[15]:


def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val


# In[16]:


rmse(volume_predict, volume_real)


# In[19]:


#mean_absolute_error(volume_real, volume_predict)


# In[42]:


y_true = [3, 3, 2, 2]
y_pred = [3, 3, 2,2]
mean_absolute_error(volume_real, volume_predict)

