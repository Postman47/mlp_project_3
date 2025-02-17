import pandas as pd
import numpy as np
import math
import os

def _get_path_to_file(data_dir ,file_path):
    '''generates path to file, needs name of wanted file and constant that decides whether
      we use it for classification set or regression set
    '''
    return os.path.normpath(os.path.join(data_dir, file_path))

def _read_csv_data(path):
    return pd.read_csv(path)

def _get_column_data(data, column_index):  
    '''give vector of values of chosen column from csv file, in our case 0=x 1=y 2=cls'''
    return data.iloc[:, column_index].values.tolist()

def _nan_to_mean(df):
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda col: col.fillna(col.mean()))
    return df

def _nan_to_zero(df):
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda col: col.fillna(0))
    return df

def _nan_to_median(df):
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda col: col.fillna(col.median()))
    return df

def _nan_to_most_frequent(df):
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(lambda col: col.fillna(col.mode()[0]))
    return df

def _normalize_data(df):
    firstColumn = df.iloc[:, 0]
    max = df.iloc[:, 1:].values.max()
    min = df.iloc[:, 1:].values.min()
    df_normalized = (df.iloc[:, 1:] - min) / (max - min)
    return pd.concat([firstColumn, df_normalized], axis=1)

def _description_process(df):
    firstColumn = df.iloc[:, 0]
    unique_val = pd.unique(df.iloc[:, 1:].values.ravel())
    value_to_index = {value: idx for idx, value in enumerate(unique_val)}
    df_mapped = df.iloc[:, 1:].map(lambda x: value_to_index[x])
    return pd.concat([firstColumn, df_mapped], axis=1)

def _sort_columns(df1, df2, df3, df4, df5, df6):
    firstColumn = df1.iloc[:, 0]
    columns = df1.iloc[:, 1:].columns
    result = pd.concat([df1.iloc[:, 1:] , df2.iloc[:, 1:], df3.iloc[:, 1:], df4.iloc[:, 1:], df5.iloc[:, 1:], df6.iloc[:, 1:]], axis=1)
    result.columns = ([f"{col}_hum" for col in columns] + [f"{col}_pre" for col in columns] + [f"{col}_tem" for col in columns]
                      + [f"{col}_wea" for col in columns] + [f"{col}_dir" for col in columns] + [f"{col}_spe" for col in columns])
    sorted_df = result.sort_index(axis=1)
    return pd.concat([firstColumn, sorted_df], axis=1)

def _wind_direction_toxy(windDirection):

    angle_in_rad = math.radians(90 - windDirection)

    x = math.sin(angle_in_rad)
    y = math.cos(angle_in_rad)

    return pd.Series([x, y])

def _encode_datetime(df):
    
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek

    # Cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Time delta
    reference_date = df['datetime'][0]
    df['days_since_ref'] = (df['datetime'] - reference_date).dt.total_seconds() / (24 * 3600)

    return df

def _process_wind_direction(df):
    xy_components = df.iloc[:, 1:].apply(lambda col: col.map(_wind_direction_toxy))
    df_1 = pd.concat([df.iloc[:, 0], xy_components], axis=1)
    # df_xy = pd.DataFrame({
    #     col: pd.Series([x[0] for x in xy_components[col]], dtype=float)
    #     for col in xy_components.columns
    # })
    # df_xy = pd.concat([df_xy, pd.DataFrame({
    #     col: pd.Series([x[1] for x in xy_components[col]], dtype=float)
    #     for col in xy_components.columns
    # })], axis=1)
    
    # # Rename columns for clarity
    # df_xy.columns = [f"{col}_x" for col in df_xy.columns[:len(df.columns)]] + \
    #   [f"{col}_y" for col in f_xy.columns[len(df.columns):]]
    return df_1

def _remove_incomplete_days(df):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['date'] = df['datetime'].dt.date
    day_counts = df['date'].value_counts()
    complete_days = day_counts[day_counts == 24].index
    filtered_df = df[df['date'].isin(complete_days)].drop(columns=['date'])
    filtered_df.reset_index(drop=True, inplace=True)
    return filtered_df

def load_data(data_dir):
    df_city_attr = _read_csv_data(_get_path_to_file(data_dir, 'city_attributes.csv'))
    ids = _get_column_data(df_city_attr, 0)
    coordinates = df_city_attr.iloc[:, 2:]

    df_hum = _normalize_data(_nan_to_most_frequent(_remove_incomplete_days(_read_csv_data(_get_path_to_file(data_dir, 'humidity.csv')))))
    df_pre = _normalize_data(_nan_to_most_frequent(_remove_incomplete_days(_read_csv_data(_get_path_to_file(data_dir, 'pressure.csv')))))
    df_tem = _normalize_data(_nan_to_most_frequent(_remove_incomplete_days(_read_csv_data(_get_path_to_file(data_dir, 'temperature.csv')))))
    df_wea = _normalize_data(_description_process(_remove_incomplete_days(_read_csv_data(_get_path_to_file(data_dir, 'weather_description.csv')))))
    df_dir = _normalize_data(_nan_to_most_frequent(_remove_incomplete_days(_read_csv_data(_get_path_to_file(data_dir, 'wind_direction.csv')))))
    df_spe = _normalize_data(_nan_to_most_frequent(_remove_incomplete_days(_read_csv_data(_get_path_to_file(data_dir, 'wind_speed.csv')))))

    datetimes = pd.DataFrame(columns=['datetime'])
    datetimes['datetime'] = pd.to_datetime(df_hum['datetime'])
    datetimes = _encode_datetime(datetimes)

    return ids, coordinates, datetimes, df_hum, df_pre, df_tem, df_wea, df_dir, df_spe

if __name__ == '__main__':
    print(load_data('./data'))


