import numpy as np
import pandas as pd
from datetime import datetime
import math

def feature_extraction(data_frame, diff_in_hour, num_days_ago,num_hours_forecasting, test_split = 0.1, validation_split = 0.2):
    '''
    This function takes as an input a data-frame that includes the following colums:
    '[Year, Month, Day, Weekday, Hour, Load] where each row represents the load at this time.'
    The output is a dataframe where each rows is a feature: 
    [Year, cos+sin(Month), cos+sin(day), cos+sin(weekday), cos+sin(hour), normalized load].
    This function also normalized the load with the assumtion that it is a gaussian, and output the mean and the std
    of the load. 
    '''
    num_hours_ahead = num_hours_forecasting - 1
    df = data_frame.copy()
    # The rows in which train val and test sets end:
    train_rows = int(len(df)*(1-test_split-validation_split))
    val_rows = int(len(df)*(1-test_split))
    test_rows = int(len(df))
    
    # normalizing the Load
    mean_value = df.loc[0:train_rows, 'Load'].mean()
    std_value = df.loc[0:train_rows, 'Load'].std()
    df['Load'] = (df['Load']-mean_value)/std_value
    
    df['Datetime'] = df['Datetime'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    
    df['Year'] = df['Datetime'].apply(lambda x: x.year)
          
    # featurs of Month_CS from Month

    df['Month_C'] = df['Datetime'].apply(lambda x: np.cos(2*math.pi*x.month/12))
    df['Month_S'] = df['Datetime'].apply(lambda x: np.sin(2*math.pi*x.month/12))

    # featurs of Day_CS from Day
    df['Day_C'] = df['Datetime'].apply(lambda x: np.cos(2*math.pi*x.day/31))
    df['Day_S'] = df['Datetime'].apply(lambda x: np.sin(2*math.pi*x.day/31))

    # featurs of Weekday_CS from Weekday
    df['Weekday_C'] = df['Datetime'].apply(lambda x: np.cos(2*math.pi*x.weekday()/7))
    df['Weekday_S'] = df['Datetime'].apply(lambda x: np.sin(2*math.pi*x.weekday()/7))

    # featurs of Hour_CS from Hour
    df['Hour_C'] = df['Datetime'].apply(lambda x: np.cos(2*math.pi*x.hour/24))
    df['Hour_S'] = df['Datetime'].apply(lambda x: np.sin(2*math.pi*x.hour/24))
    
    if not diff_in_hour == 1:
        df['Min_C'] = df['Datetime'].apply(lambda x: np.cos(2*math.pi*x.minute/60))
        df['Min_S'] = df['Datetime'].apply(lambda x: np.sin(2*math.pi*x.minute/60))
    
    #features drop
    df = df.drop(columns=['Datetime'], axis=1)
    
    # features of consumption from num_days_back
    n_consumption_features = int(24 * num_days_ago/ diff_in_hour)
    for i in range(1,n_consumption_features+1):
        df[f'{i * diff_in_hour}_hrs_ago'] = df['Load'].shift(i)
    
    # features of consumption from num_hours_ahead
    if num_hours_ahead != 0:
        n_consumption_features = int(num_hours_ahead/ diff_in_hour)
        for i in range(1,n_consumption_features+1):
            df[f'{i * diff_in_hour}_hrs_ahead'] = df['Load'].shift(-i)
        
    df = df.rename(columns={'Load': '0_hrs_ahead'})
    
    # remove rows with nan's
    df = df[~df.isna().any(axis=1)]
    features = df.copy()
    #splitting the data into train, validation and test:
        
    ago_cols = [c for c in df.columns if c.endswith('_hrs_ago')]
    x_train_seq = df.loc[0:train_rows, ago_cols].values
    x_val_seq = df.loc[train_rows:val_rows, ago_cols].values
    x_test_seq = df.loc[val_rows: test_rows, ago_cols].values
    
    ahead_cols = [c for c in df.columns if c.endswith('_hrs_ahead')]
    y_train = df.loc[0:train_rows, ahead_cols].values
    y_val = df.loc[train_rows:val_rows, ahead_cols].values
    y_test = df.loc[val_rows: test_rows, ahead_cols].values
    
    df = df.drop(ago_cols + ahead_cols, axis=1)

    x_train_dt = df.loc[0:train_rows].values
    x_val_dt = df.loc[train_rows:val_rows].values
    x_test_dt = df.loc[val_rows: test_rows].values
    
    data = {'x_train_seq': np.array(x_train_seq),
            'x_train_dt': np.array(x_train_dt),
            'x_val_seq': np.array(x_val_seq),
            'x_val_dt': np.array(x_val_dt),
            'x_test_seq': np.array(x_test_seq),
            'x_test_dt': np.array(x_test_dt),
            'y_train': np.array(y_train),
            'y_val': np.array(y_val),
            'y_test': np.array(y_test),
            'mean_value_load': mean_value,
            'std_value_load': std_value
           }
    
    return features, data