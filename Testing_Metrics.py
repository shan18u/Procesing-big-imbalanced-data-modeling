
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from google.colab import drive
import matplotlib.pyplot as plt

# Define a function to train and evaluate a model with a specific scaler
def train_and_evaluate(scaler):
    # Scale 'Time' and 'Amount' features
    data['scaled_time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
    data['scaled_amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    # Drop original 'Time' and 'Amount' features
    data_scaled = data.drop(['Time', 'Amount'], axis=1)
    
    # Split data into features (X) and target variable (y)
    X = data_scaled.drop('Class', axis=1)
    y = data_scaled['Class']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # Apply SMOTE to balance the data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    # Train logistic regression model
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_resampled, y_resampled)
    
    # Predict on test set and calculate metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    # Return performance metrics
    return accuracy, precision, recall

# Mount Google Drive to access data
drive.mount('/content/drive')

# Load data, remove duplicates and missing values, and re-label Class 0 to -1
data = pd.read_csv('/content/drive/MyDrive/creditcard copy.csv')
data = data.drop_duplicates()
data = data.dropna()
data['Class'] = data['Class'].replace({0: -1})

# Define scalers to be evaluated
scalers = {
    'RobustScaler': RobustScaler(),
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler()
}

# Create dictionaries to save results
accuracy_results = {}
precision_results = {}
recall_results = {}

# Train and evaluate a model for each scaler
for scaler_name, scaler in scalers.items():
    accuracy, precision, recall = train_and_evaluate(scaler)
    
    # Save results
    accuracy_results[scaler_name] = accuracy
    precision_results[scaler_name] = precision
    recall_results[scaler_name] = recall
    
    # Print results
    print(f"{scaler_name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}\n")

# Convert dictionaries to Series for easier plotting
accuracy_series = pd.Series(accuracy_results)
precision_series = pd.Series(precision_results)
recall_series = pd.Series(recall_results)

# Create subplots
fig, ax = plt.subplots(3, 1, figsize=(10, 15))

# Plot results
accuracy_series.plot(kind='bar', ax=ax[0], title='Accuracy')
precision_series.plot(kind='bar', ax=ax[1], title='Precision')
recall_series.plot(kind='bar', ax=ax[2], title='Recall')

plt.tight_layout()
plt.show()
