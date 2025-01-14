import pandas as pd

def sortColumns(df1, df2, df3, df4, df5, df6):
    columns = df1.iloc[:, 1:].columns
    result = pd.concat([df1 , df2.iloc[:, 1:], df3.iloc[:, 1:], df4.iloc[:, 1:], df5.iloc[:, 1:], df6.iloc[:, 1:]], axis=1)
    result.columns = ([f"{col}_hum" for col in columns] + [f"{col}_pre" for col in columns] + [f"{col}_tem" for col in columns]
                      + [f"{col}_wea" for col in columns] + [f"{col}_dir" for col in columns] + [f"{col}_spe" for col in columns])
    return result.sort_index(axis=1)