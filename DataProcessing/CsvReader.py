import pandas as pd
import os

class CsvReader:

    def getPathToFile(self, file_name):  # generates path to file, needs name of wanted file and constant that decides whether we use it for classification set or regression set
        return os.path.normpath(os.path.join(os.getcwd() + os.sep + 'Data' + os.sep + file_name))

    def readCsvData(self, path):
        return pd.read_csv(path)

    def getColumnData(self, data,
                      column_index):  # give vector of values of chosen column from csv file, in our case 0=x 1=y 2=cls
        return data.iloc[:, column_index].values.tolist()
