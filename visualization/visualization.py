import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.metrics import mean_squared_error, f1_score
import math
import scipy.stats as st
import plotly.express as px
import plotly.graph_objs as go

def visualization_df(df, chart_title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close price'))
    fig.update_layout(
        title=chart_title,
        xaxis_title="Date",
        yaxis_title="Close price", width=1350, height=500
    )
    fig.show()


def visualization_train(train_mse_loss):
    fig = px.histogram(train_mse_loss, nbins=50, histnorm='probability density')
    fig.update_layout(xaxis_title='Loss', yaxis_title='Density', title='Distribution of Train MSE Loss', width=1350, height=500)
    fig.show()


def visualization_test(test_score_df):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=test_score_df.index, y=test_score_df.loss, mode='lines', name='loss'))
    fig.add_trace(go.Scatter(x=test_score_df.index, y=test_score_df.threshold, mode='lines', name='threshold'))

    fig.update_layout(title='Test Loss vs Threshold', xaxis_title='Date', yaxis_title='Loss', width=1350, height=500)
    fig.show()


def visualization_anomaly(test,anomalies,TIME_STEPS,scaler):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test[TIME_STEPS:].index,
    y=scaler.inverse_transform(test[TIME_STEPS:].close.values.reshape(1,-1)).reshape(-1),
    mode='lines', name='Close price'))
    fig.add_trace(go.Scatter(x=anomalies.index,
    y=scaler.inverse_transform(anomalies.close.values.reshape(1,-1)).reshape(-1),
    mode='markers', marker=dict(color='red', size=10),
    name='Anomaly'))
    fig.update_layout(title='Close price vs Anomalies',
    xaxis_title='Date',
    yaxis_title='Close price',
    legend=dict(x=0, y=1, traceorder="normal"), width=1350, height=500)
    fig.show()

