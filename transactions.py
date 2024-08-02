import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import plotly.express as px

from visualization.data_balancing import data_balancing

df = pd.read_csv('creditcard.csv')

print(df.head())
print(df.info())
print(df['Class'].value_counts())

fig = px.histogram(df, x='Class', title='Class Distribution')
fig.show()

df.fillna(df.mean(), inplace=True)

scaler = StandardScaler()
df['scaled_amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['scaled_time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.drop(['Time', 'Amount'], axis=1, inplace=True)

df = df[['scaled_amount', 'scaled_time'] + [col for col in df.columns if col not in ['scaled_amount', 'scaled_time', 'Class']] + ['Class']]

X = df.drop('Class', axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train_resampled, y_train_resampled = data_balancing(X_train, y_train)

anom_detector = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
anom_detector.fit(X_train)

y_pred_anom = anom_detector.predict(X_test)
y_pred_anom = np.where(y_pred_anom == -1, 1, 0)  # 1 for anomalies, 0 for normal

classifier = Sequential()
classifier.add(Dense(64, activation='relu', input_dim=X_train_resampled.shape[1]))
classifier.add(Dropout(0.5))
classifier.add(Dense(32, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(1, activation='sigmoid'))
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train_resampled, y_train_resampled, epochs=50, batch_size=32, validation_split=0.2, shuffle=True)

y_pred_cls = classifier.predict(X_test)
y_pred_cls = np.where(y_pred_cls > 0.5, 1, 0)

print('Confusion Matrix for Anomaly Detection:')
print(confusion_matrix(y_test, y_pred_anom))
print('Classification Report for Anomaly Detection:')
print(classification_report(y_test, y_pred_anom))

print('Confusion Matrix for Fraud Detection:')
print(confusion_matrix(y_test, y_pred_cls))
print('Classification Report for Fraud Detection:')
print(classification_report(y_test, y_pred_cls))

accuracy = accuracy_score(y_test, y_pred_cls)
precision = precision_score(y_test, y_pred_cls)
recall = recall_score(y_test, y_pred_cls)
f1 = f1_score(y_test, y_pred_cls)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')

