import pandas as pd

def sortColumns(df1, df2, df3, df4, df5, df6):
    columns = df1.iloc[:, 1:].columns
    result = pd.concat([df1 , df2.iloc[:, 1:], df3.iloc[:, 1:], df4.iloc[:, 1:], df5.iloc[:, 1:], df6.iloc[:, 1:]], axis=1)
    result.columns = ([f"{col}_1" for col in columns] + [f"{col}_2" for col in columns] + [f"{col}_3" for col in columns]
                      + [f"{col}_4" for col in columns] + [f"{col}_5" for col in columns] + [f"{col}_6" for col in columns])
    return result.sort_index(axis=1)