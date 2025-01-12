import pandas as pd

class NanToMean:
    def __init__(self):
        pass

    def NanToMean(self, df):
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda col: col.fillna(col.mean()))
        return df