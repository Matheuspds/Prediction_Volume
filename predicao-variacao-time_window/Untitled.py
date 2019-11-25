
# coding: utf-8

# In[12]:


import pandas as pd
path = "data_after_process/"
path_final = "data_final/"


# In[13]:


def add_feature_clima(df_volume):
    df_weather = pd.read_csv("data_after_process/feature_clima.csv")[["date", "hour", "precipitation", "rel_humidity"]]
    df_volume = df_volume.merge(df_weather, on=["date", "hour"], how="left")
    
    return df_volume


# In[14]:


def ampm(x):
    if (x <= 12 ):
        return 1
    return 0


# In[15]:


def generate_features(df_volume):
    df_volume["am_pm"] = df_volume["hour"].apply(lambda x: ampm(x))
    df_volume = add_feature_clima(df_volume)
    
    return df_volume


# In[16]:


df_test1 = pd.read_csv(path+"test_filter_0.csv" )
df_test2 = pd.read_csv(path+"test_filter_5.csv" )
df_test3 = pd.read_csv(path+"test_filter_10.csv" )
df_test4 = pd.read_csv(path+"test_filter_15.csv" )
df_train1 = pd.read_csv(path+"train_filter_0.csv")
df_train2 = pd.read_csv(path+"train_filter_5.csv")
df_train3 = pd.read_csv(path+"train_filter_10.csv")
df_train4 = pd.read_csv(path+"train_filter_15.csv")


# In[17]:


generate_features(df_test1).to_csv(path_final+"test_final_0.csv",index=False)
generate_features(df_test2).to_csv(path_final+"test_final_5.csv",index=False)
generate_features(df_test3).to_csv(path_final+"test_final_10.csv",index=False)
generate_features(df_test4).to_csv(path_final+"test_final_15.csv",index=False)
generate_features(df_train1).to_csv(path_final+"train_final_0.csv",index=False)
generate_features(df_train2).to_csv(path_final+"train_final_5.csv",index=False)
generate_features(df_train3).to_csv(path_final+"train_final_10.csv",index=False)
generate_features(df_train4).to_csv(path_final+"train_final_15.csv",index=False)

