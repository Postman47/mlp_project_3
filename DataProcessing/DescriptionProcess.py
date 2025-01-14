import pandas as pd


class DescriptionProcess(object):
    def __init__(self):
        pass

    def descriptionProcess(self, df):
        firstColumn = df.iloc[:, 0]
        unique_val = pd.unique(df.iloc[:, 1:].values.ravel())
        value_to_index = {value: idx for idx, value in enumerate(unique_val)}
        df_mapped = df.iloc[:, 1:].map(lambda x: value_to_index[x])
        return pd.concat([firstColumn, df_mapped], axis=1)