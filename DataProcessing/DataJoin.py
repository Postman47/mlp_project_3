import pandas as pd

import DataProcessing.CsvReader as reader
import DataProcessing.WindDirectionProcess as wdp
from DataProcessing.WindDirectionProcess import WindDirectionProcess
import DataProcessing.NanToMean as ntm
from DataProcessing.DescriptionProcess import DescriptionProcess


class DataJoin:

    def joinData(self):
        path_hum = reader.CsvReader.getPathToFile(None, f"humidity.csv")
        path_pre = reader.CsvReader.getPathToFile(None, f"pressure.csv")
        path_tem = reader.CsvReader.getPathToFile(None, f"temperature.csv")
        path_wea = reader.CsvReader.getPathToFile(None, f"weather_description.csv")
        path_dir = reader.CsvReader.getPathToFile(None, f"wind_direction.csv")
        path_spe = reader.CsvReader.getPathToFile(None, f"wind_speed.csv")

        df_hum = reader.CsvReader.readCsvData(None, path_hum)
        df_pre = reader.CsvReader.readCsvData(None, path_pre)
        df_tem = reader.CsvReader.readCsvData(None, path_tem)
        df_wea = reader.CsvReader.readCsvData(None, path_wea)
        df_dir = reader.CsvReader.readCsvData(None, path_dir)
        df_spe = reader.CsvReader.readCsvData(None, path_spe)

        df_wea = DescriptionProcess.descriptionProcess(None, df_wea)

        df_dir = ntm.NanToMean.NanToMean(None, df_dir)
        wdp = WindDirectionProcess()
        df_dir = wdp.processWindDirection(df_dir)



        return pd.concat([df_hum , df_pre.iloc[:, 1:], df_tem.iloc[:, 1:], df_wea.iloc[:, 1:], df_dir.iloc[:, 1:], df_spe.iloc[:, 1:]], axis=1)