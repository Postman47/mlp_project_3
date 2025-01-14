import pandas as pd

import DataProcessing.CsvReader as reader
import DataProcessing.WindDirectionProcess as wdp
from DataProcessing.WindDirectionProcess import WindDirectionProcess
from DataProcessing.NanToMean import nanToMean as ntm
from DataProcessing.DescriptionProcess import DescriptionProcess
from DataProcessing.NormalizeData import normalizeData
from DataProcessing.SortColumns import sortColumns


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

        df_hum = ntm(df_hum)
        df_pre = ntm(df_pre)
        df_tem = ntm(df_tem)
        df_dir = ntm(df_dir)
        df_spe = ntm(df_spe)

        df_hum = normalizeData(df_hum)
        df_pre = normalizeData(df_pre)
        df_tem = normalizeData(df_tem)
        df_spe = normalizeData(df_spe)

        wdp = WindDirectionProcess()
        df_dir = wdp.processWindDirection(df_dir)

        return sortColumns(df_hum, df_pre, df_tem, df_wea, df_dir, df_spe)