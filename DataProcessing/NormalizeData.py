import pandas as pd

def normalizeData(df):
    firstColumn = df.iloc[:, 0]
    df_normalized = (df.iloc[:, 1:] - df.iloc[:, 1:].min()) / (df.iloc[:, 1:].max() - df.iloc[:, 1:].min())
    return pd.concat([firstColumn, df_normalized], axis=1)