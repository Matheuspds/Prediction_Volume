
# coding: utf-8

# In[1]:


import pandas as pd
path = "data_after_process_valores_anteriores/"
path_final = "data_final/"


# In[2]:


def add_feature_clima(df_volume):
    df_weather = pd.read_csv("data_after_process/feature_clima.csv")
    df_volume = pd.merge(df_volume, df_weather, how = 'left', on = ['date','hour'])
    
    return df_volume


# In[3]:


def ampm(x):
    if (x <= 12 ):
        return 1
    return 0


# In[4]:


def generate_features(df_volume):
    df_volume["am_pm"] = df_volume["hour"].apply(lambda x: ampm(x))
    df_volume = add_feature_clima(df_volume)
    
    return df_volume


# In[5]:


df_test1 = pd.read_csv(path+"test0_valor_anterior.csv" )
df_test2 = pd.read_csv(path+"test5_valor_anterior.csv" )
df_test3 = pd.read_csv(path+"test10_valor_anterior.csv" )
df_test4 = pd.read_csv(path+"test15_valor_anterior.csv" )
df_train1 = pd.read_csv(path+"train0_valor_anterior.csv")
df_train2 = pd.read_csv(path+"train5_valor_anterior.csv")
df_train3 = pd.read_csv(path+"train10_valor_anterior.csv")
df_train4 = pd.read_csv(path+"train15_valor_anterior.csv")


# In[6]:


generate_features(df_test1).to_csv(path_final+"test_final_0.csv",index=False)
generate_features(df_test2).to_csv(path_final+"test_final_5.csv",index=False)
generate_features(df_test3).to_csv(path_final+"test_final_10.csv",index=False)
generate_features(df_test4).to_csv(path_final+"test_final_15.csv",index=False)
generate_features(df_train1).to_csv(path_final+"train_final_0.csv",index=False)
generate_features(df_train2).to_csv(path_final+"train_final_5.csv",index=False)
generate_features(df_train3).to_csv(path_final+"train_final_10.csv",index=False)
generate_features(df_train4).to_csv(path_final+"train_final_15.csv",index=False)

