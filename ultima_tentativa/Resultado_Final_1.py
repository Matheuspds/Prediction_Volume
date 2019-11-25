
# coding: utf-8

# In[198]:


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


# In[323]:


df_train = pd.read_csv('data_process_final/treino_final.csv')
df_teste = pd.read_csv('data_process_final/teste_final.csv')


# In[324]:


df_remove = df_train.loc[(df_train['day'] >= 1) & (df_train['day'] <= 7) ]

df_train = df_train.drop(df_remove.index)


# In[325]:


del df_train['volume_proximo']
del df_teste['volume_proximo']


# In[326]:


del df_train['volume_proximo_2']
del df_teste['volume_proximo_2']


# In[327]:


def adiciona_media_desvio_por_dia(df):
    df['media_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])["volume"].transform(np.mean)
    df['desvio_padrao_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])["volume"].transform(np.std)
    df['desvio_padrao_hora_dia'].fillna(df.groupby(['direction', 'tollgate_id', 'day', 'hour'])["volume"].transform(np.mean), inplace=True)
    df['min_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])['volume'].transform(np.min)
    df['max_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])['volume'].transform(np.max)
    df['mediana_volume_hora_dia'] = df.groupby(['direction', 'tollgate_id', 'day', 'hour'])['volume'].transform(np.median)
    return df


# In[329]:


df_train = adiciona_media_desvio_por_dia(df_train)
df_teste= adiciona_media_desvio_por_dia(df_teste)


# In[330]:


def adiciona_media_desvio_por_janela_dia_semana_1(df):
    df['time'] = pd.to_datetime(df['time'], format = '%Y-%m-%d %H:%M:%S')
    df['date'] = df['time'].dt.date
    mask = (df['time'] > '2016-09-18 00:00:00') & (df['time'] < '2016-09-26 00:00:00')
    df = df.loc[mask]
    df['min_volume_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.min)
    df['max_volume_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.max)
    df['mediana_volume_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.median)
    df['media_volume_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean)
    df['desvio_padrao_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.std)
    df['desvio_padrao_weekday'].fillna(df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
    return df

def adiciona_media_desvio_por_janela_dia_semana_2(df):
    
    df['time'] = pd.to_datetime(df['time'], format = '%Y-%m-%d %H:%M:%S')
    df['date'] = df['time'].dt.date
    mask2 = (df['time'] > '2016-09-18 00:00:00') & (df['time'] < '2016-10-10 00:00:00')
    df = df.loc[mask2]
    df['min_volume_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.min)
    df['max_volume_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.max)
    df['mediana_volume_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.median)
    df['media_volume_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean)
    df['desvio_padrao_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.std)
    df['desvio_padrao_weekday'].fillna(df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
    
    return df

def adiciona_media_desvio_por_janela_dia_semana_3(df):
    
    df['time'] = pd.to_datetime(df['time'], format = '%Y-%m-%d %H:%M:%S')
    df['date'] = df['time'].dt.date
    mask3 = (df['time'] > '2016-09-18 00:00:00') & (df['time'] < '2016-10-17 00:00:00')
    df = df.loc[mask3]
    df['min_volume_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.min)
    df['max_volume_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.max)
    df['mediana_volume_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.median)
    df['media_volume_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean)
    df['desvio_padrao_weekday'] = df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.std)
    df['desvio_padrao_weekday'].fillna(df.groupby(['am_pm', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
    return df

def adiciona_media_desvio_por_janela_dia_semana_4(df):   
    df['time'] = pd.to_datetime(df['time'], format = '%Y-%m-%d %H:%M:%S')
    df['date'] = df['time'].dt.date
    mask4 = (df['time'] > '2016-09-18 00:00:00') & (df['time'] < '2016-10-25 00:00:00')
    df = df.loc[mask4]
    df['min_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.min)
    df['max_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.max)
    df['mediana_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.median)
    df['media_volume_weekday'] = df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean)
    df['desvio_padrao_weekday'] = df.groupby(['time_window', 'week','direction', 'tollgate_id'])["volume"].transform(np.std)
    df['desvio_padrao_weekday'].fillna(df.groupby(['time_window', 'week', 'direction', 'tollgate_id'])["volume"].transform(np.mean), inplace=True)
    
    
    return df


# In[331]:


df_train_a_25 =adiciona_media_desvio_por_janela_dia_semana_1(df_train)
df_train_a_10 = adiciona_media_desvio_por_janela_dia_semana_2(df_train)
df_train_a_17 = adiciona_media_desvio_por_janela_dia_semana_3(df_train)
df_train_a_24 = adiciona_media_desvio_por_janela_dia_semana_4(df_train)


# In[332]:


df_train_list = [df_train_a_25, df_train_a_10, df_train_a_17, df_train_a_24]
df_train_para_agregar = pd.concat(df_train_list)


# In[335]:


df_teste['min_volume_weekday'] = df_train_a_24['min_volume_weekday']
df_teste['max_volume_weekday'] = df_train_a_24['max_volume_weekday']
df_teste['mediana_volume_weekday'] = df_train_a_24['mediana_volume_weekday']
df_teste['media_volume_weekday'] = df_train_a_24['media_volume_weekday']
df_teste['desvio_padrao_weekday'] = df_train_a_24['desvio_padrao_weekday']


# In[ ]:


#media do volume do dia naquela tollgate naquela direcao


# In[336]:


def medidas_volume_tollgate_direction_am_pm(df):
    df['min_volume_dia_am_pm'] = df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.min)
    df['max_volume_dia_am_pm'] = df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.max)
    df['mediana_volume_dia_am_pm'] = df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.median)
    df['media_volume_dia_am_pm'] = df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.mean)
    df['desvio_padrao_am_pm'] = df.groupby(['day','direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.std)
    df['desvio_padrao_am_pm'].fillna(df.groupby(['day', 'direction', 'tollgate_id', 'am_pm'])["volume"].transform(np.mean), inplace=True)
    return df


# In[337]:


df_train_para_agregar = medidas_volume_tollgate_direction_am_pm(df_train_para_agregar)
df_teste = medidas_volume_tollgate_direction_am_pm(df_teste)


# In[338]:


def feature_format():
    #pd_volume_train = pd_volume_train.set_index(['time'])
    #pd_volume_test = pd_volume_test.set_index(['time'])
    #volume_train = v_train.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #volume_test = v_test.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    #print(volume_train)                
    x = pd.Series(df_train_para_agregar['time_window'].unique())
    s = pd.Series(range(len(x)),index = x.values)
    df_train_para_agregar['window_n'] = df_train_para_agregar['time_window'].map(s)
    df_teste['window_n'] = df_teste['time_window'].map(s)
#        print vol_test.tail()
    #volume_train['weekday'] = v_train['weekday']
    #volume_test['weekday'] = v_test['weekday']
    
    feature_train = df_train_para_agregar.drop('volume', axis = 1)
    feature_test = df_teste.drop('volume',axis = 1)
    values_train = df_train_para_agregar['volume'].values
    values_test = df_teste['volume'].values
    
    return feature_train, feature_test, values_train, values_test


# In[339]:


feature_train, feature_test, values_train, values_test = feature_format()


# In[340]:


feature_train.columns


# In[342]:


regressor_cubic = RandomForestRegressor(n_estimators=500, max_depth=10, oob_score=True)


# In[349]:


regressor_cubic.fit(feature_train[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday']], values_train)


# In[350]:


y_pred = regressor_cubic.predict(feature_test[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday']])


# In[354]:


mean_absolute_percentage_error(values_test, y_pred)


# In[352]:


rmse = sqrt(mean_squared_error(y_pred, values_test))
rmse


# In[117]:


# Create the parameter grid based on the results of random search 
param_grid = {
    'max_depth': [int(x) for x in np.linspace(10, 110, num = 11)],
    'max_features': [None],
    'n_estimators': [200, 300, 500, 800, 1200, 1500, 1800]
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


# In[118]:


grid_search.fit(feature_train[['tollgate_id','direction', 'hour', 'week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia']], values_train)


# In[30]:


grid_search.best_params_


# In[33]:


best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, feature_test[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_test)


# In[36]:


print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))


# In[111]:


regressor_cubic = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=10, oob_score=True)


# In[112]:


regressor_cubic.fit(feature_train[[1,2,3,'direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia']], values_train)


# In[113]:


y_pred = regressor_cubic.predict(feature_test[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia']])


# In[114]:


mean_absolute_percentage_error(values_test, y_pred)


# In[99]:


rmse = sqrt(mean_squared_error(y_pred, values_test))
rmse


# In[ ]:


##ADABOOSTING REGRESSOR


# In[61]:


def ADABooster(param_grid, n_jobs): 
    estimator = AdaBoostRegressor() 
    cv = ShuffleSplit(feature_train[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']].shape[0], test_size=0.2) 
    classifier = GridSearchCV(estimator=estimator, cv=4, param_grid=param_grid, n_jobs=n_jobs) 
    classifier.fit(feature_train[['tollgate_id', 'direction','week', 'am_pm', 'volume_anterior', 'volume_anterior_2', 'avg_vol_dia_semana', 'desvio_padrao', 'window_n']], values_train) 
    print ("Best Estimator learned through GridSearch")
    print (classifier.best_estimator_)
    return cv, classifier.best_estimator_


# In[62]:


param_grid={'n_estimators':[100, 300, 500, 800, 1200], 
            'learning_rate': [0.1, 0.05, 0.01, 0.005], 
            'loss':['linear', 'square', 'exponential']}
n_jobs=4 

cv,best_est=ADABooster(param_grid, n_jobs)


# In[355]:


regr = AdaBoostRegressor(n_estimators=1200, learning_rate=0.005, loss='exponential')


# In[356]:


regr.fit(feature_train[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday']], values_train) 


# In[123]:


regr.feature_importances_


# In[357]:


y_pred_ada = regr.predict(feature_test[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday']])


# In[132]:


regr.score(feature_train[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia']], values_train)


# In[358]:


mean_absolute_percentage_error(values_test, y_pred_ada)


# In[359]:


rmse = sqrt(mean_squared_error(values_test, y_pred_ada))
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


# In[353]:


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


# In[364]:




regressor_cubic_g = GradientBoostingRegressor(n_estimators=250, learning_rate=0.05, min_samples_split=15,
                                                min_samples_leaf=2, max_depth=5, subsample=0.8,
                                                random_state=10, loss="lad")


# In[365]:


regressor_cubic_g.fit(feature_train[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday']], values_train)
yhat = regressor_cubic_g.predict(feature_test[['tollgate_id','direction','week', 'weekend','volume_anterior', 'volume_anterior_2', 'am_pm', 'window_n', 'desvio_padrao_hora_dia', 'mediana_volume_hora_dia', 'max_volume_hora_dia', 'media_volume_hora_dia', 'min_volume_hora_dia', 'media_volume_dia_am_pm', 'max_volume_dia_am_pm', 'mediana_volume_dia_am_pm', 'min_volume_dia_am_pm', 'desvio_padrao_am_pm', 'media_volume_weekday', 'desvio_padrao_weekday']])


# In[366]:


mean_absolute_percentage_error(values_test, yhat)


# In[367]:


rmse = sqrt(mean_squared_error(values_test, yhat))
rmse

