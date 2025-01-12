import pandas as pd
import math

class WindDirectionProcess(object):
    def __init__(self):
        pass

    def windDirectionToxy(self, windDirection):

        angle_in_rad = math.radians(90 - windDirection)

        x = math.sin(angle_in_rad)
        y = math.cos(angle_in_rad)

        return pd.Series([x, y])

    def processWindDirection(self, df):
        xy_components = df.iloc[:, 1:].apply(lambda col: col.map(self.windDirectionToxy))
        df_1 = pd.concat([df.iloc[:, 0], xy_components], axis=1)
        # df_xy = pd.DataFrame({
        #     col: pd.Series([x[0] for x in xy_components[col]], dtype=float)
        #     for col in xy_components.columns
        # })
        # df_xy = pd.concat([df_xy, pd.DataFrame({
        #     col: pd.Series([x[1] for x in xy_components[col]], dtype=float)
        #     for col in xy_components.columns
        # })], axis=1)
        #
        # # Rename columns for clarity
        # df_xy.columns = [f"{col}_x" for col in df_xy.columns[:len(df.columns)]] + [f"{col}_y" for col in
        #                                                                            df_xy.columns[len(df.columns):]]
        return df_1
