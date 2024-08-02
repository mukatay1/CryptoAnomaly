<h1>Data Preparation</h1>
<ul>
    <li>Read the datasets into pandas DataFrames.</li>
    <li>Displayed structure and summary statistics to understand the data.</li>
    <li>Normalized necessary features using StandardScaler.</li>
    <li>Created additional time-series structured datasets for LSTM using specified time steps.</li>
</ul>
<h1>Model Development</h1>
<ul>
    <li>Trained an Isolation Forest model on the training set from the credit card data.</li>
    <li>Detected anomalies as potential frauds using LTSM.</li>
    <li>For fraud detection, build a neural network with Dense and Dropout layers.</li>
    <li>For price prediction, construct an LSTM model designed to handle sequential data.</li>
</ul>


<h1>Model Training and Prediction</h1>
<ul>
    <li>Fited models on the respective training datasets.</li>
    <li>Used validation splits to monitor performance and avoid overfitting.</li>
    <li>Predictd on the test datasets.</li>
    <li>For anomaly detection, adjust labels to denote anomalies.</li>
</ul>

<h1>Evaluation and Visualization</h1>
<ul>
    <li>Calculated accuracy, precision, recall, F1 score, and confusion matrices for the fraud detection model.</li>
    <li>Computed MSE, RMSE, and MAE for the Ethereum prediction model.</li>
    <li>Used Plotly and Matplotlib to visualize class distributions, actual vs. predicted values, and anomalies.</li>
    <li>Labeled data points as anomalies based on the set thresholds and visualize them.</li>
</ul>



