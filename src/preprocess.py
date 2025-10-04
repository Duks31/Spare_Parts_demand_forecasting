import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """

    Preprocess the input DataFrame by dropping unnecessary columns and renaming year values.

    Parameters:
    df (pd.DataFrame): Input DataFrame to preprocess.

    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    # split the week column into year, month, day
    df[['day', 'month', 'year']] = df['week'].str.split("/", expand=True)

    # Drop unnecessary columns
    df = df.drop(['week','store_id', 'total_price', 'is_featured_sku', 'is_display_sku'], axis=1)
    
    # Rename year values
    df['year'] = df['year'].replace({'11': '2023', '12': '2024', '13': '2025'})

    # Convert year, month, day to integers
    df['date'] = pd.to_datetime(df[['year','month','day']])
    
    # df = df.drop(['day', 'month', 'year'], axis=1)

    df = df.sort_values(['sku_id','date']).reset_index(drop=True)


    return df

def create_features(data: pd.DataFrame, lags=[1,2,4], windows=[3,6]) -> pd.DataFrame:

    df = data.copy()

    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('sku_id')['cost'].shift(lag)

    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby('sku_id')['cost'].shift(1).rolling(window).mean()

    # Date-based features
    df['month'] = df['date'].dt.month
    df['quarter'] = df['date'].dt.quarter
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week.astype(int)
    df['date'] = pd.to_datetime(df[['year','month','day']])
    return df