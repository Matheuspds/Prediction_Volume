
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


# In[2]:


pd_volume_train = pd.read_csv('processed_train_volume2.csv')


# In[4]:


pd_volume_train.head()


# In[5]:


pd_volume_train['time'] =  pd.to_datetime(pd_volume_train['time'] , format='%Y-%m-%d %H:%M:%S')
#pd_volume_train = pd_volume_train.set_index(['time_window'])

# 车流量
pd_volume_train = pd_volume_train.groupby([pd.Grouper(freq='20T', key='time'), 'tollgate_id', 'direction', 'time_window', 'date', 'hour', 'weekday']).size()       .reset_index().rename(columns = {0:'volume'})


# In[6]:


pd_volume_train.head()


# In[36]:


baseline = RandomForestRegressor()


# In[37]:


baseline.get_params().keys()


# In[10]:


metricas = ['explained_variance', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2']


# In[35]:


# Create the parameter grid based on the results of random search 
hyper = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}


# In[41]:


grid_search = GridSearchCV(estimator = baseline, param_grid = hyper, cv = 3, n_jobs = -1, verbose = 2)


# In[42]:


def feature_format():
    v_train = pd.read_csv('dados_treino_volume_com_valor_anterior.csv')
    v_test = pd.read_csv('dados_teste_volume_com_valor_anterior.csv')
    #pd_volume_train = pd_volume_train.set_index(['time'])
    #pd_volume_test = pd_volume_test.set_index(['time'])
    #volume_train = v_train.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #volume_test = v_test.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #print(volume_train)                
    x = pd.Series(v_train['time_window'].unique())
    s = pd.Series(range(len(x)),index = x.values)
    v_train['window_n'] = v_train['time_window'].map(s)
    v_test['window_n'] = v_test['time_window'].map(s)
#        print vol_test.tail()
    #volume_train['weekday'] = v_train['weekday']
    #volume_test['weekday'] = v_test['weekday']
    
    feature_train = v_train.drop('volume', axis = 1)
    feature_test = v_test.drop('volume',axis = 1)
    values_train = v_train['volume'].values
    values_test = v_test['volume'].values
    
    return feature_train, feature_test, values_train, values_test


# In[43]:


feature_train, feature_test, values_train, values_test = feature_format()


# In[44]:


grid_search.fit(feature_train[['window_n','tollgate_id', 'direction', 'weekday', 'volume_anterior', 'volume_anterior_2','media_volume', 'desvio_padrao']], values_train)


# In[46]:


grid_search.best_params_


# In[48]:


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy


# In[50]:


best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, feature_test[['window_n','tollgate_id', 'direction', 'weekday', 'volume_anterior', 'volume_anterior_2','media_volume', 'desvio_padrao']], values_test)


# In[51]:


print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))


# In[21]:


pd.set_option('max_columns',200)
pd.DataFrame(meu_primeiro_grid.cv_results_)

