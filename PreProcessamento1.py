
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


# In[2]:


# Descrição das features:
# time           datetime  Data e Hora em que o veículo passa pelo pedágio;
# tollgate_id    string    Identificador do pedágio;
# direction      string    0: entra na rodovia pelo pedágio; 1: sai da rodovia pelo pedágio;
# vehicle_model  int       Um número que indica a capacidade do veículo;
# has_etc        string    Indica se o veículo possui ou não o sistema ETC; 0 - NÃO, 1 - SIM
# vehicle_type   string    0: veículo de passageiro; 1: veículo de carga
# weekday        int       Representa os dias da semana
# weekend        int       1: Para quando for fim de semana; 0: Para quando não for fim de semana


# In[15]:


volume_df = pd.read_csv("dataset/volume(table 6)_test1.csv")
volume_df.head()


# In[5]:


#Caracteristicas dos dados
volume_df.info()
volume_df.describe()


# In[16]:


#Retirando os valores nulos da coluna vehicle_type pelo modelo do veículo.
    #No vehicle_type indica 0 para veículo de passageiros e 1 para carga.
    #Poderíamos verificar a partir do modelo do veiculo, para veiculo com capacidade de até 4
    #Ficou definido que sera para passageiro, sendo maior que 4 será veiculo de carga


# In[17]:


#volume_df['vehicle_type'] = volume_df['vehicle_model'].apply(lambda x: 0 if x < 5 else 1)


# In[18]:


volume_df.head()


# In[19]:


volume_df.tail()


# In[20]:


#Função que será usada para obter a janela de tempo de 20 minutos
def get_timewindow(t):
        time_window = 20
        if t.minute < time_window:
            window = [time(t.hour, 0), time(t.hour,20)]
        elif t.minute < time_window*2:
            window = [time(t.hour, 20), time(t.hour, 40)]
        else:
            try:
                window = [time(t.hour, 40), time(t.hour + 1, 0)]
            except ValueError:
                window = [time(t.hour, 40), time(0,0,0)]
        s_window = '[' + str(window[0]) + ',' + str(window[1]) + ')'
        return s_window

def get_hour(t):
        return t.hour


# In[21]:


#Ajustando o formato da coluna time
volume_df['time'] = pd.to_datetime(volume_df['time'], format = '%Y-%m-%d %H:%M:%S')

#Representam os feriados
holiday1 = [pd.Timestamp('2016-10-1'), pd.Timestamp('2016-10-8')]
holiday2 = [pd.Timestamp('2016-9-15'), pd.Timestamp('2016-9-18')]

#Adiciona valores para os dias da semana
volume_df['weekday'] = volume_df['time'].dt.dayofweek + 1

#Classificar cada atributo de time aplicando a janela de tempo de vinte minutos
volume_df['t'] = volume_df['time'].dt.time

#Adicionando valores para saber se é referente a um fim de semana ou não
volume_df['weekend'] = volume_df['weekday'].apply(lambda x: 0 if x < 6 else 1)

volume_df['hour'] = volume_df['t'].apply(get_hour)

volume_df['date'] = volume_df['time'].dt.date

volume_df['holiday'] = volume_df['time'].between(holiday1[0],holiday1[1])                            | volume_df['time'].between(holiday2[0],holiday2[1])

volume_df['time_window'] = volume_df['t'].apply(get_timewindow)
del volume_df['t']

volume_df.head()


# In[22]:


#Salvando dados de treino
volume_df.to_csv('processed_test_volume2.csv', index = False)


# In[46]:


#Fazendo o mesmo processo para os dados de teste


# In[41]:


pd_volume_train = pd.read_csv('processed_training_volume2.csv')
pd_volume_test = pd.read_csv('processed_test_volume2.csv')
pd_volume_train = pd_volume_train.set_index(['time'])
pd_volume_test = pd_volume_test.set_index(['time'])
volume_train = pd_volume_train.groupby(['time_window','tollgate_id','direction', 'hour', 'date']).size().reset_index().rename(columns = {0:'volume'})
volume_test = pd_volume_test.groupby(['time_window','tollgate_id','direction', 'hour', 'date']).size().reset_index().rename(columns = {0:'volume'})

x = pd.Series(volume_train['time_window'].unique())
s = pd.Series(range(len(x)),index = x.values)
volume_train['window_n'] = volume_train['time_window'].map(s)
volume_test['window_n'] = volume_test['time_window'].map(s)
#volume_test.tail()
#volume_train.tail()
#volume_train.head()
volume_test.head()


# In[100]:


def feature_format():
    pd_volume_train = pd.read_csv('processed_train_volume2.csv')
    pd_volume_test = pd.read_csv('processed_test_volume2.csv')
    #pd_volume_train = pd_volume_train.set_index(['time'])
    #pd_volume_test = pd_volume_test.set_index(['time'])
    volume_train = pd_volume_train.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
    volume_test = pd_volume_test.groupby(['time_window','tollgate_id','direction','date', 'hour']).size().reset_index().rename(columns = {0:'volume'})
                    
    x = pd.Series(volume_train['time_window'].unique())
    s = pd.Series(range(len(x)),index = x.values)
    volume_train['window_n'] = volume_train['time_window'].map(s)
    volume_test['window_n'] = volume_test['time_window'].map(s)
#        print vol_test.tail()
    volume_train['weekday'] = pd_volume_train['weekday']
    volume_test['weekday'] = pd_volume_test['weekday']
    
    feature_train = volume_train.drop('volume', axis = 1)
    feature_test = volume_test.drop('volume',axis = 1)
    values_train = volume_train['volume'].values
    values_test = volume_test['volume'].values
    
    return feature_train, feature_test, values_train, values_test


# In[104]:


pd_volume_train = pd.read_csv('processed_train_volume2.csv')
pd_volume_train.head()


# In[101]:


feature_train, feature_test, values_train, values_test = feature_format()


# In[103]:


print(feature_train)


# In[93]:


feature_train[['window_n', 'tollgate_id', 'direction', 'weekday']].head


# In[84]:


print(values_train)


# In[85]:


print(feature_test)


# In[86]:


rng = np.random.RandomState(1)
regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 15),
                         n_estimators=20, random_state = rng)


# In[87]:


regr.fit(feature_train[['window_n', 'tollgate_id', 'direction', 'weekday']], values_train)

y_pred = regr.predict(feature_test[['window_n', 'tollgate_id', 'direction', 'weekday']])

mape = np.mean(np.abs((y_pred - values_test)/values_test))

#print (feature_test)
print (mape)

#print (y_pred,values_test)


# In[26]:


x_test.head()


# In[27]:


y_test


# In[77]:


#Função que calcula a média quadrática
def rmse(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)
    return rmse_val


# In[90]:


#Função que calcula o MAPE
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[89]:


#Executando algoritmo de regressão linear
model1 = LinearRegression()
model1.fit(feature_train[['window_n', 'tollgate_id', 'direction', 'weekday']], values_train)
pred_y1 = model1.predict(feature_test[['window_n', 'tollgate_id', 'direction', 'weekday']])
pred_y1


# In[81]:


#Algoritmo Regressão Linear com Gradiente Descendente
class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return self.net_input(X)


# In[89]:


model2 = LinearRegressionGD(eta = 0.0009, n_iter = 40)
model2.fit(x_train, y_train)
pred_y2 = model2.predict(x_test)


# In[32]:


#Algoritmo Regressão Linear com Gradiente Descendente Estocrástico
model3 = SGDRegressor(eta0=0.0009, max_iter=40)
model3.fit(x_train, y_train)
pred_y3 = model3.predict(x_test)


# In[91]:


mean_absolute_percentage_error(values_test, pred_y1)


# In[102]:


print(  rmse(pred_y1, y_test),"\n", #Algoritmo 1
        rmse(pred_y2, y_test),"\n", #Algoritmo 2
        rmse(pred_y3, y_test) #Algoritmo 3
     )


# In[33]:


mean_squared_error(pred_y3, y_test)


# In[40]:


rng = np.random.RandomState(1)
regr = AdaBoostRegressor(DecisionTreeRegressor(max_depth = 21),
                         n_estimators=11, random_state = rng)

regr.fit(x_train, y_train)

y_pred = regr.predict(x_test)

mape = np.mean(np.abs((y_pred - y_test)/y_test))

print (mape)

