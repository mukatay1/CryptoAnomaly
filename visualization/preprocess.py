def preprocess_data(df):
    df = df.dropna()
    df = df[(df['close'] > df['close'].quantile(0.01)) & (df['close'] < df['close'].quantile(0.99))]
    return df

