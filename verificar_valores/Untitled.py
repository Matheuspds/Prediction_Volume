
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import time
from sklearn.metrics import mean_absolute_error


# In[2]:


pd_test_real = pd.read_csv("test_real.csv")


# In[9]:


pd_test_real.count()


# In[4]:


pd_pred = pd.read_csv("resultado_finalverificar.csv")


# In[10]:


pd_pred.count()


# In[6]:


volume_real = pd_test_real['volume'].values
volume_predict = pd_pred['volume'].values


# In[7]:


#Função que calcula o MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[8]:


mean_absolute_percentage_error(volume_real, volume_predict)

