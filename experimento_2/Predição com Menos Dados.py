
# coding: utf-8

# In[100]:


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import math
import random
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.cross_validation import ShuffleSplit, train_test_split
from sklearn.metrics import mean_squared_error
path_final = 'dados_finais/'
from sklearn import preprocessing
from sklearn import svm
import os
get_ipython().run_line_magic('matplotlib', 'inline')
import bokeh.sampledata
from bokeh.io import output_notebook, show
from bokeh.plotting import figure,show
from datetime import datetime as dt


# In[66]:


df_train = pd.read_csv('treino_final.csv')


# In[69]:


df_outliers = df_train[(df_train['day'] >= 1) & (df_train['day'] <= 7)]


# In[136]:


#fig = figure(plot_width=400, plot_height=400)
plt.bar(df_outliers['day'], df_outliers['volume'])
#fig.circle(dataset['day'], dataset['volume'], fill_color="yellow", size=10)
plt.rcParams['figure.figsize'] = (11,8)
plt.xlabel('Dias do mês')
plt.ylabel('Volume do tráfego')
plt.title('Volume do fluxo de tráfego durante as comemorações do Dia Nacional')
plt.ylim(100, 300)
plt.show()


# In[150]:


df_normal = df_train[(df_train['day'] >= 8) & (df_train['day'] <= 14)]


# In[158]:


#fig = figure(plot_width=400, plot_height=400)
df_train['time'] = pd.to_datetime(df_train['time'], format = '%Y-%m-%d %H:%M:%S')
plt.bar(df_normal['day'], df_normal['volume'])
#fig.circle(dataset['day'], dataset['volume'], fill_color="yellow", size=10)
plt.rcParams['figure.figsize'] = (11,8)
plt.xlabel('Dias do mês')
plt.ylabel('Volume do tráfego')
plt.title('Volume do fluxo de tráfego na semana após as comemorações do Dia Nacional')
plt.ylim(0, 200)
plt.show()


# In[84]:


df_train['time'] = pd.to_datetime(df_train['time'], format = '%Y-%m-%d %H:%M:%S')
df_train['date'] = df_train['time'].dt.date


# In[85]:


df_train


# In[108]:


df_train['date'] = pd.to_datetime(df_train['date'])


df_normal_2 = df_train[(df_train['date'] >= '2016-09-23') & (df_train['date'] <= '2016-09-29')]


# In[139]:


plt.bar(df_normal_2['day'], df_normal_2['volume'])
#fig.circle(dataset['day'], dataset['volume'], fill_color="yellow", size=10)
plt.rcParams['figure.figsize'] = (11,8)
plt.xlabel('Dias do mês')
plt.ylabel('Volume do tráfego')
plt.title('Volume do fluxo de tráfego na semana anterior as comemorações do Dia Nacional')
plt.ylim(100, 200)
plt.show()


# In[169]:


df_day_normal_23 = df_train[(df_train['date'] == '2016-10-10')]


# In[170]:


plt.bar(df_day_normal_23['hour'], df_day_normal_23['volume'])
#fig.circle(dataset['day'], dataset['volume'], fill_color="yellow", size=10)
plt.rcParams['figure.figsize'] = (11,8)
plt.xlabel('Dias do mês')
plt.ylabel('Volume do tráfego')
plt.title('Volume do fluxo de tráfego por hora no dia 10 de Outubro.')
plt.ylim(0, 200)
plt.show()


# In[128]:


df_day_normal_fim_de_semana_8 = df_train[(df_train['date'] == '2016-10-01')]


# In[145]:


plt.bar(df_day_normal_fim_de_semana_8['hour'], df_day_normal_fim_de_semana_8['volume'])
#fig.circle(dataset['day'], dataset['volume'], fill_color="yellow", size=10)
plt.rcParams['figure.figsize'] = (11,8)
plt.xlabel('Dias do mês')
plt.ylabel('Volume do tráfego')
plt.title('Volume do fluxo de tráfego por hora no dia 1 de Outubro durante as comemorações do Dia Nacional')
plt.ylim(0, 300)
plt.show()


# In[67]:


X_df = pd.DataFrame(data=df_train)
X_df['LogMedHouseVal'] = values_train
_ = X_df.hist(column=['day'])


# In[92]:


# the histogram of the data
volume = df_train['volume'].value_counts().values
time = df_train['volume'].value_counts().index
plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.bar(time, volume, ec = "k", alpha = .6, color = "royalblue")
plt.subplot(1,2,2)


# In[31]:


dataset = df_train[ df_train['day'] >= 28 ]


# In[41]:


dataset['day'].value_counts().plot(kind='bar');
#plt.rcParams['figure.figsize'] = (11,11)


# In[35]:


dataset['volume'].values


# In[ ]:


param_grid = {
    'max_depth': [1,2,3,4,5,6,7,8,9,10,15,20],
    'max_features': [None],
    'n_estimators': [200, 300, 500, 800, 1200, 1500]
}
# Create a based model
gbrt = GradientBoostingRegressor()
# Instantiate the grid search model
grid_search_gbrt = GridSearchCV(estimator = gbrt, param_grid = param_grid, 
                          cv = 4, n_jobs = -1, verbose = 2)


# In[ ]:


grid_search_gbrt.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'max_volume_weekday', 'mediana_volume_weekday', 'min_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[71]:


#fig = figure(plot_width=400, plot_height=400)
plt.bar(dataset['day'], dataset['volume'])
#fig.circle(dataset['day'], dataset['volume'], fill_color="yellow", size=10)
plt.rcParams['figure.figsize'] = (11,8)
plt.show()

