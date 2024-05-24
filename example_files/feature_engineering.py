import pandas as pd
import numpy as np

def transform_altitude(df:pd.DataFrame)-> pd.DataFrame:
    df["altitude_mean_log"] = np.log(df["altitude_mean_meters"])
    df = df.drop(['altitude_mean_meters', ], axis=1)
    return df
    

#'Unnamed: 0' and Quakers
def drop_column(df:pd.DataFrame, col_name: str) -> pd.DataFrame:
    df = df.drop([col_name], axis=1)
    return df

def fill_missing_values(df:pd.DataFrame) -> pd.DataFrame:
    altitude_low_meters_mean=1500.3684210526317
    altitude_high_meters_mean=1505.6315789473683
    altitude_mean_log_mean=7.0571530664031155
    df["altitude_low_meters"] = df["altitude_low_meters"].fillna(altitude_low_meters_mean)
    df["altitude_high_meters"] = df["altitude_high_meters"].fillna(altitude_high_meters_mean)
    df["altitude_mean_log"] = df["altitude_mean_log"].fillna(altitude_mean_log_mean)
    return df



def windspeed (df:pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    df_new["windspeed"] = np.sqrt(np.power(df_new["v_component_of_wind_10m_above_ground"], 2) + np.power(df_new["u_component_of_wind_10m_above_ground"],2))
    return df_new[["windspeed"]]


def target_class(df:pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    df_new['PM_warning'] =df_new['target'].apply(lambda x: 1 if x >= 50 else 0)
    return df_new[["PM_warning"]]


def target_previous(df:pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    df_new["target_previous"] = df_new.groupby("Place_ID")["target"].shift()
    return df_new[["target_previous"]]



def forw_fill_na (df:pd.DataFrame) -> pd.DataFrame:
    df_new = df.copy()
    filled = df_new.fillna(method="ffill", limit=1)
    return filled



def adjusted_r_squared(r_squared, X):
    adjusted_r2 = 1 - ((1 - r_squared) * (len(X) - 1) / (len(X) - X.shape[1] - 1))
    return adjusted_r2