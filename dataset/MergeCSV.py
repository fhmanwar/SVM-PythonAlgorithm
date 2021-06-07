import pandas as pd
import os, glob
# import matplotlib.pyplot as plt
# import numpy as np
# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# df = pd.read_csv('reg.csv', usecols=['Total', 'BITS', 'HydroCoco', 'Love_Juice', 'Carrot_S'])
df1 = pd.read_csv('ekspedisi_JNE_FIX.csv')
df2 = pd.read_csv('ekspedisi_JNT.csv')
df3 = pd.read_csv('ekspedisi_Sicepat.csv')
# df4 = pd.read_csv('data.csv')

# print(df) # cek All Data with table
# print(df.info()) # cek Label Column
# print(df.isnull().sum()) # cek Label Column with sum data
# print(df.shape) # cek row X Column
# print(df.describe()) # cek favorite count and retweet count

# print(df1.info())
# print(df2.info())
# print(df3.info())
# print(df4.describe())

def CleanText(data):
    # Removing Duplicate if any
    df = data.drop_duplicates()
    df = data.reset_index(drop=True)

    # Clean text
    len(df[df['clean_text'].isnull()==True])
    pd.set_option('display.max_colwidth', None)
    df[df['clean_text'].isnull()==True]['original_text']

    # Removing Observation
    df = df.dropna(subset=['clean_text'])
    df = df.reset_index(drop=True)
    return df

# print(df1)
# print(CleanText(df1))

jne = CleanText(df1)
jnt = CleanText(df2)
sicepat = CleanText(df3)

# print(jne.head(300))

# df_merged = pd.concat([jne.head(300), jnt.head(300), sicepat.head(300)], ignore_index=True)
df_merged = pd.concat([sicepat.head(300)], ignore_index=True)
# print(df_merged)
df_merged.to_csv("sicepat300.csv")


# path = "./"
# all_files = glob.glob(os.path.join(path, "ekspedisi_*.csv"))
# # print(all_files)

# df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
# # print(df_from_each_file)

# df_merged = pd.concat(df_from_each_file, ignore_index=True)
# # print(df_merged)
# df_merged.to_csv("merged.csv")



# df1 = pd.DataFrame({
#     'Kode Barang': y,
# })
# data_output = pd.concat([df, df1], axis=1)
# data_output.to_csv('data.csv', index=False, encoding='utf-8')
