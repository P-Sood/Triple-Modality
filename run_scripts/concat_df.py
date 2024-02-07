import pandas as pd

PATH = "../data/"

df = pd.DataFrame()
for i in range(0,21):
    try:
        tmp = pd.read_pickle(f"{PATH}URFUNNY_df_sub_{i}.pkl")
        df = pd.concat([df, tmp]) 
    except:
        continue
# print(df)
df.to_pickle(f"{PATH}urfunny_small_combo.pkl")