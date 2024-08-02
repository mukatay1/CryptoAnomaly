import requests
import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
import plotly.graph_objs as go
import matplotlib.pyplot as plt

from visualization.visualization import visualization_df, visualization_train, visualization_test, visualization_anomaly
from visualization.preprocess import preprocess_data

TF_ENABLE_ONEDNN_OPTS = 0

url = 'https://api.coingecko.com/api/v3/coins/ethereum/market_chart'
headers = {"accept": "application/json"}
params = {'days': '365', 'vs_currency': 'usd'}

response = requests.get(url, params=params, headers=headers)
data = response.json()
df = pd.DataFrame(data['prices'], columns=['date', 'close'])
df['date'] = pd.to_datetime(df['date'], unit='ms')
df.set_index('date', inplace=True)


def data_engineering(df):
    df = preprocess_data(df)
    train_size = int(len(df) * 0.80)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    scaler = StandardScaler()
    scaler.fit(train[['close']])
    train['close'] = scaler.transform(train[['close']])
    test['close'] = scaler.transform(test[['close']])

    return train, test, scaler


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


def train_test_split(train, test, TIME_STEPS):
    x_train, y_train = create_dataset(train[['close']], train['close'], TIME_STEPS)
    x_test, y_test = create_dataset(test[['close']], test['close'], TIME_STEPS)
    return x_train, x_test, y_train, y_test


def get_model(input_shape):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(units=128, input_shape=input_shape))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.RepeatVector(n=input_shape[0]))
    model.add(keras.layers.LSTM(units=128, return_sequences=True))
    model.add(keras.layers.Dropout(rate=0.2))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=input_shape[1])))
    model.compile(loss='mae', optimizer='adam')
    return model


def display_actual_predicted(df, predictions, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df.values.reshape(-1), mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=df.index, y=predictions.reshape(-1), mode='lines', name='Predicted'))
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price', legend=dict(x=0, y=1, traceorder="normal"),
                      width=1350, height=500)
    fig.show()


def visualization_pred(df, x_test_pred, test, TIME_STEPS):
    fig, ax = plt.subplots(figsize=(17, 5))
    ax.plot(test.index[TIME_STEPS:], test[TIME_STEPS:].close, label='Actual')
    ax.plot(test.index[TIME_STEPS:], x_test_pred[:, -1], label='Predicted')
    ax.legend()
    ax.set_title('Actual vs Predicted Prices')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    plt.show()


TIME_STEPS = 30
list_df = [df]

for i, df in enumerate(list_df, start=1):
    print('Start')
    df = df[['close']]
    train, test, scaler = data_engineering(df)
    x_train, x_test, y_train, y_test = train_test_split(train, test, TIME_STEPS)

    model = get_model((x_train.shape[1], x_train.shape[2]))
    model.summary()

    history = model.fit(x_train, y_train, epochs=100, batch_size=16, validation_split=0.1, shuffle=False)

    x_train_pred = model.predict(x_train)
    train_mse_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

    x_test_pred = model.predict(x_test)
    test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)

    visualization_train(train_mse_loss)

    THRESHOLD = np.percentile(test_mae_loss, 94)
    test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
    test_score_df['loss'] = test_mae_loss
    test_score_df['threshold'] = THRESHOLD
    test_score_df['anomaly'] = test_score_df['loss'] > test_score_df['threshold']
    test_score_df['close'] = test[TIME_STEPS:].close

    anomalies = test_score_df[test_score_df['anomaly'] == True]

    visualization_test(test_score_df)
    visualization_pred(df, x_test_pred, test, TIME_STEPS)
    visualization_anomaly(test, anomalies, TIME_STEPS, scaler)

    mse = np.mean(np.square(x_test_pred - x_test))
    rmse = np.sqrt(mse)
    precision, recall, thresholds = precision_recall_curve(test_score_df['anomaly'], test_score_df['loss'])
    f1_score = 2 * (precision * recall) / (precision + recall)
    best_f1_score = np.max(f1_score)
    mae = np.mean(np.abs(x_test_pred - x_test))
    std_dev = np.std(test_mae_loss)
    num_anomalies = len(anomalies)

    print(
        f'Dataframe {i}: MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, F1 Score: {best_f1_score:.4f}, Number of Anomalies: {num_anomalies}')
    print('_______________________________________________')
