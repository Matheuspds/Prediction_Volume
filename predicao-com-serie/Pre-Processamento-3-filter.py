
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


#Função que adiciona uma nova coluna para saber se o horário am ou pm, identificado pelos valores
#1 e 0 respectivamente.
def manha_ou_tarde(x):
    if (x <= 12 ):
        return 1
    return 0


# In[4]:


#Função que diz se a hora é mais especifica ou não. Verifica dois horários dos quais podem ter bem mais fluxo
#durante todo o dia
def calc_period_num(x):
    if x == 8 or x == 17:
        return 1
    return 0


# In[17]:


def generate_train(filname):
    df_volume = pd.read_csv(filname)

    df_volume["time"] = pd.to_datetime(df_volume["time"])
    df_volume = df_volume.sort_values(['tollgate_id', 'direction', 'time'])
    df_volume["am_pm"] = df_volume["hour"].apply(lambda x: manha_ou_tarde(x))

    for shift_num in range(0, 6):
        f2 = lambda x: x.values[shift_num]

        df_volume[str(shift_num)] = df_volume[["tollgate_id", "direction", "volume", "date", "am_pm"]].groupby(
            ["tollgate_id", "direction", "date", "am_pm"]).transform(f2)

    df_volume = df_volume[
        (df_volume["hour"] == 8) |
        (df_volume["hour"] == 9) |
        (df_volume["hour"] == 17) |
        (df_volume["hour"] == 18)]

    df_volume["period_num"] = df_volume["hour"].apply(lambda x: calc_period_num(x))
    df_volume["period_num"] = df_volume["period_num"] + df_volume["miniute"].apply(lambda x: x / 20)

    df_volume["hour1"] = df_volume["hour"].apply(lambda x: x / 3 * 3)
    df_weather = pd.read_csv("data_after_process/feature_clima.csv")[["date", "hour", "precipitation", "rel_humidity"]]
    df_volume = df_volume.merge(df_weather, on=["date", "hour"], how="left")

    df_volume = df_volume.drop("hour1", axis=1)


    return df_volume


# In[10]:


def combined_train():
    path = "data_after_process/"
    df1 = generate_train(path+"train_filter_0.csv")
    df1["volume"] = df1["volume"].replace(0, 1)
    df1.to_csv("train.csv", index=False)

    df2 = generate_train(path+"train_filter_5.csv")
    df2["volume"] = df2["volume"].replace(0, 1)
    df2.to_csv("train1.csv", index=False)

    df3 = generate_train(path+"train_filter_10.csv")
    df3["volume"] = df3["volume"].replace(0, 1)
    df3.to_csv("train2.csv", index=False)

    df4 = generate_train(path+"train_filter_15.csv")
    df4["volume"] = df4["volume"].replace(0, 1)
    df4.to_csv("train3.csv", index=False)


# In[11]:


def get_test():
    path = "data_after_process/"
    df1 = generate_train(path+"test_filter_0.csv")
    df1.to_csv("test2.csv", index=False)


# In[18]:


combined_train()
get_test()


# In[24]:


path = "data_after_process/"
df_hue = pd.read_csv("train3.csv")


# In[25]:


df_hue.head()

