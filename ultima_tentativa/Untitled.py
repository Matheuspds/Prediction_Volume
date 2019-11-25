
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from datetime import time
import matplotlib.pyplot as pplot
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import math
import random
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.cross_validation import ShuffleSplit, train_test_split


# In[6]:


df_train = pd.read_csv('data_process_final/treino_final.csv')
df_teste = pd.read_csv('data_process_final/teste_final.csv')


# In[51]:


df_teste.count()


# In[7]:


df_remove = df_train.loc[(df_train['day'] >= 1) & (df_train['day'] <= 7) ]

df_train = df_train.drop(df_remove.index)


# In[24]:


#df_remove = df_train.loc[(df_train['hour'] >= 7)  & (df_train['hour'] <= 10) ]

#df_train = df_train.drop(df_remove.index)


# In[26]:


#df_remove = df_train.loc[(df_train['hour'] >= 15)  & (df_train['hour'] <= 17) ]

#df_train = df_train.drop(df_remove.index)


# In[48]:


df_train.count()


# In[7]:


df_train.isnull().sum()


# In[8]:


def feature_format():
    #pd_volume_train = pd_volume_train.set_index(['time'])
    #pd_volume_test = pd_volume_test.set_index(['time'])
    #volume_train = v_train.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #volume_test = v_test.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #print(volume_train)                
    x = pd.Series(df_train['time_window'].unique())
    s = pd.Series(range(len(x)),index = x.values)
    df_train['window_n'] = df_train['time_window'].map(s)
    df_teste['window_n'] = df_teste['time_window'].map(s)
#        print vol_test.tail()
    #volume_train['weekday'] = v_train['weekday']
    #volume_test['weekday'] = v_test['weekday']
    
    feature_train = df_train.drop('volume', axis = 1)
    feature_test = df_teste.drop('volume',axis = 1)
    values_train = df_train['volume'].values
    values_test = df_teste['volume'].values
    
    return feature_train, feature_test, values_train, values_test


# In[9]:


feature_train, feature_test, values_train, values_test = feature_format()


# In[10]:


feature_train.isnull().sum()


# In[11]:


feature_test.isnull().sum()


# In[15]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[16]:


rf = RandomForestRegressor()


# In[17]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 4, verbose=2, random_state=42, n_jobs = -1)


# In[18]:


rf_random.fit(feature_train[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_train)


# In[19]:


rf_random.best_params_


# In[20]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# In[22]:


base_model = RandomForestRegressor(n_estimators = 10, random_state = 42)
base_model.fit(feature_train[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_train)
base_accuracy = evaluate(base_model, feature_test[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_test)


# In[23]:


best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, feature_test[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_test)


# In[24]:


print('Improvement of {:0.2f}%.'.format( 100 * (random_accuracy - base_accuracy) / base_accuracy))


# In[31]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
    #colocar um intervalo mais logico. Treino teste e validacao (tira uma parte do treino para validacao, e os
    # e eu nao posso fazer isso pra proxima janela de treino)
    #volume proximo nao pode ser utilizado
    # a media de dias da semana anterior
    # max altura, max numero de features, n_estimadores (100, 200, 300, 400)
    # max altura (80, 90,100, 110)
    # n_features deixa o numero de features fixo que eu tenho passa o auto
    # random_state tirar do decision tree
    #fazer isso
    #falar primeiro do decision tree antes mesmo do random forest
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 4, n_jobs = -1, verbose = 2)


# In[32]:


grid_search.fit(feature_train[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_train)


# In[33]:


grid_search.best_params_


# In[34]:


best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, feature_test[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_test)


# In[35]:


print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))


# In[10]:


regressor_cubic = RandomForestRegressor(n_estimators = 200, max_depth=110, bootstrap=True, max_features=3, min_samples_leaf=4, min_samples_split=8)


# In[11]:


regressor_cubic.fit(feature_train[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_train)


# In[12]:


y_pred = regressor_cubic.predict(feature_test[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']])


# In[15]:


mean_absolute_percentage_error(values_test, y_pred)


# In[16]:


rmse = sqrt(mean_squared_error(values_test, y_pred))
rmse


# In[50]:


rng = np.random.RandomState(1)
regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 50),
                         n_estimators=300, random_state = rng)


# In[51]:


regr.fit(feature_train[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_train)

y_pred = regr.predict(feature_test[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']])


# In[52]:


rmse = sqrt(mean_squared_error(values_test, y_pred))
rmse


# In[14]:


def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


# In[58]:


mean_absolute_percentage_error(values_test, y_pred)


# In[1]:


#GRADIENT BOOSTING


# In[22]:


#Fazendo o GBRT COM PARAMETROS MAIS SIMPLES


# In[24]:


gbrt=GradientBoostingRegressor(n_estimators=100) 
gbrt.fit(feature_train[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_train)

y_pred_gbrt=gbrt.predict(feature_test[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']])


# In[25]:


mean_absolute_percentage_error(values_test, y_pred_gbrt)


# In[35]:


def GradientBooster(param_grid, n_jobs): 
    estimator = GradientBoostingRegressor() 
    cv = ShuffleSplit(feature_train[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']].shape[0], n_iter=10, test_size=0.2) 
    classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=param_grid, n_jobs=n_jobs) 
    classifier.fit(feature_train[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_train) 
    print ("Best Estimator learned through GridSearch")
    print (classifier.best_estimator_)
    return cv, classifier.best_estimator_


# In[36]:


param_grid={'n_estimators':[100], 
            'learning_rate': [0.1], 
            'max_depth':[6], 
            'min_samples_leaf':[3],
            'max_features':[1.0]}
n_jobs=4 

cv,best_est=GradientBooster(param_grid, n_jobs)


# In[38]:


print ("Best Estimator Parameters") 
print("---------------------------" )
print ("n_estimators: %d" %best_est.n_estimators) 
print ("max_depth: %d" %best_est.max_depth) 
print ("Learning Rate: %.1f" %best_est.learning_rate) 
print ("min_samples_leaf: %d" %best_est.min_samples_leaf) 
print ("max_features: %.1f" %best_est.max_features) 

print ("Train R-squared: %.2f" %best_est.score(feature_train[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_train) ) 


# In[40]:


estimator = best_est
estimator.fit(feature_train[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_train) 


# In[43]:


print ("Train R-squared: %.2f" %estimator.score(feature_train[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_train)  )

print ("Test R-squared: %.2f" %estimator.score(feature_test[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_test)) 


# In[44]:


estimator = GradientBoostingRegressor(n_estimators=best_est.n_estimators, max_depth=best_est.max_depth, learning_rate=best_est.learning_rate, min_samples_leaf=best_est.min_samples_leaf, max_features=best_est.max_features)


# In[45]:


estimator.fit(feature_train[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_train)


# In[46]:


y_pred_gbrt_oficial = estimator.predict(feature_test[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'volume_proximo', 'volume_proximo_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']])


# In[48]:


mean_absolute_percentage_error(values_test, y_pred_gbrt_oficial)


# In[49]:


rmse = sqrt(mean_squared_error(values_test, y_pred_gbrt_oficial))
rmse

