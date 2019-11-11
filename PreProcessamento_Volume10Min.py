
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


# In[ ]:


#Função que será usada para obter a janela de tempo de 10 minutos
def get_timewindow(t):
        time_window = 10
        if t.minute < time_window:
            window = [time(t.hour, 0), time(t.hour,10)]
        elif t.minute < time_window*2:
            window = [time(t.hour, 10), time(t.hour, 20)]
        elif t.minute < time_window*3:
            window = [time(t.hour, 20), time(t.hour, 30)]
        elif t.minute < time_window*4:
            window = [time(t.hour, 30), time(t.hour, 40)]
        elif t.minute < time_window*5:
            window = [time(t.hour, 40), time(t.hour, 50)]
        elif t.minute < time_window*6:
            window = [time(t.hour, 50), time(t.hour, 60)]
        elif t.minute < time_window*7:
            window = [time(t.hour, 60), time(t.hour, 60)]
        elif t.minute < time_window*6:
            window = [time(t.hour, 50), time(t.hour, 60)]
        elif t.minute < time_window*6:
            window = [time(t.hour, 50), time(t.hour, 60)]
        else:
            try:
                window = [time(t.hour, 40), time(t.hour + 1, 0)]
            except ValueError:
                window = [time(t.hour, 40), time(0,0,0)]
        s_window = '[' + str(window[0]) + ',' + str(window[1]) + ')'
        return s_window

def get_hour(t):
        return t.hour

