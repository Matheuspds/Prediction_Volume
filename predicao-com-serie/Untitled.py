
# coding: utf-8

# In[40]:


import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import random


# In[41]:


df_test = pd.read_csv("test2.csv")

df_train0 = pd.read_csv("train.csv")
df_train1 = pd.read_csv("train1.csv")
df_train2 = pd.read_csv("train2.csv")
df_train3 = pd.read_csv("train3.csv")
df_train_list = [df_train0,df_train1, df_train2, df_train3]


# In[42]:


def feature_transform_split(key, data):
    # data = remove_exception(data)

    data["precipitation"] = data[["precipitation"]].fillna(value=0)
    data["rel_humidity"] = data[["rel_humidity"]].fillna(value=50)

    data["precipitation"] = data["precipitation"].apply(lambda x: x > 0)
    data["rel_humidity"] = data["rel_humidity"].apply(lambda x: x > 90)

    data = data.drop("precipitation", axis=1)
    # data = data.drop("rel_humidity", axis= 1)


    # data["sum"] = data["0"] + data["1"] + data["2"] + data["3"] + data["4"] + data["5"]

    data = pd.concat([data, pd.get_dummies(data['period_num'])], axis=1)
    data = data.drop("period_num", axis=1)

    data = pd.concat([data, pd.get_dummies(data['holiday'])], axis=1)
    data = data.drop("holiday", axis=1)
    #
    # data = pd.concat([data, pd.get_dummies(data['first_last_workday'])], axis=1)
    data = data.drop("first_last_workday", axis=1)

    data = data.drop("day_of_week", axis=1)

    if (key == 1):
        data = pd.concat([data, pd.get_dummies(data['tollgate_id'])], axis=1)
        # data["tollgate_id1"] = data['tollgate_id']
        data["direction1"] = data['direction']
    return data


# In[44]:


#df_hue = pd.concat(df_train_list)
#df_hue.to_csv("result/result_split_rf_huehue"+"huehue"+".csv", index=False)
random.shuffle(df_train_list)
df_train = pd.concat(df_train_list)

#df_ts = pd.read_csv("ts_feature2_simple.csv")
df_date = pd.read_csv("date.csv")

df_train = df_train.merge(df_date, on="date", how="left")
#df_train = df_train.merge(df_ts, on=["tollgate_id", "hour", "miniute", "direction"], how="left")

df_test = df_test.merge(df_date, on="date", how="left")
#df_test = df_test.merge(df_ts, on=["tollgate_id", "hour", "miniute", "direction"], how="left")

df_train_grouped = df_train.groupby(["tollgate_id", "direction"])
df_test_grouped = df_test.groupby(["tollgate_id", "direction"])
result = []
oob = []
for key, train_data in df_train_grouped:
    test_data = df_test_grouped.get_group(key)
    len_train = len(train_data)
    train_data = train_data.append(test_data)[train_data.columns.tolist()]
    train_data = feature_transform_split(key, train_data)


    regressor_cubic = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=10, oob_score=True)

    train_data = pd.DataFrame.reset_index(train_data)
    train_data = train_data.drop("index", axis=1)
    y = train_data.iloc[:len_train - 1, :]["volume"]


    x = train_data.iloc[:len_train - 1, 8:]
    x1 = train_data.iloc[len_train:, 8:]
    regressor_cubic.fit(x, y)
    print(x1)
    resultado_obtido = regressor_cubic.predict(x1)

    #df_h = test_data
    #df_h.to_csv("result/result_split_rf_huehue"+"teste"+".csv", index=False)

test_data["volume"] = resultado_obtido
result.append(test_data[['tollgate_id', 'time_window', 'direction', 'volume']])


# In[45]:


df_result = pd.concat(result, axis=0)
df_result.to_csv("resultado_final"+str(np.mean(df_result["volume"]))+".csv", index=False)


# In[47]:


df_test_real = pd.read_csv("data_after_process/test_0.csv")

#df_train0 = pd.read_csv("train.csv")
#df_train1 = pd.read_csv("train1.csv")


# In[48]:


df_test_real.head()

