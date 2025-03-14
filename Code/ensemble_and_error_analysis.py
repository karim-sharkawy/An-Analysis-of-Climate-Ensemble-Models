### ensemble model & error analysis

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


import shap # explainability
import matplotlib.pyplot as plt # visualizations

from google.colab import drive # Feel free to use whatever environment
drive.mount('/content/drive') # Make sure to remove google.colab and drive mounts and change paths

csv_path = '/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Data/climate_data_preprocessed.csv'
df_klima = pd.read_csv(csv_path)

# Need to make the date the index everytime :)
df_klima['DATE'] = pd.to_datetime(df_klima['DATE'], format='ISO8601')
df_klima.set_index('DATE', inplace=True)
df_klima.sort_index(inplace=True) # Sort DF by date

df_klima.head(3)

# Load saved models
rfr_model_path = "/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Models/random_forest_climate_model.pkl"
lstm_model_path = "/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Models/lstm_climate_model.h5"

best_rfr = joblib.load(rfr_model_path)
model_lstm = load_model(lstm_model_path, custom_objects={'mse': MeanSquaredError()})

"""# Ensemble Model

Evaluation
"""

# Simple Baseline: Predict the mean of training labels
baseline_pred = np.full_like(y_test, y_train.mean())

# Compute RMSE for Baseline
baseline_mse = mean_squared_error(y_test, baseline_pred)
baseline_rmse = baseline_mse ** 0.5
baseline_mae = mean_absolute_error(y_test, baseline_pred)
baseline_r2 = r2_score(y_test, baseline_pred)

# RF Model Predictions
rf_pred = best_rfr.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_pred)
rf_rmse = rf_mse ** 0.5
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# LSTM Model Predictions
lstm_pred = model_lstm.predict(test_generator)
y_test_lstm_actual = y_test_lstm[len(y_test_lstm) - len(lstm_pred):]  # Match lengths
lstm_mse = mean_squared_error(y_test_lstm_actual, lstm_pred)
lstm_rmse = lstm_mse ** 0.5
lstm_mae = mean_absolute_error(y_test_lstm_actual, lstm_pred)
lstm_r2 = r2_score(y_test_lstm_actual, lstm_pred)

# Print Comparison
print(f"Baseline Model RMSE: {baseline_rmse:.4f}, MAE: {baseline_mae:.4f}, R²: {baseline_r2:.4f}")
print(f"Random Forest RMSE: {rf_rmse:.4f}, MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")
print(f"LSTM RMSE: {lstm_rmse:.4f}, MAE: {lstm_mae:.4f}, R²: {lstm_r2:.4f}")

"""Visualization improvements of models compared to baseline"""

###
rmse_values = [baseline_rmse, rf_rmse, lstm_rmse]
model_names = ['Baseline', 'Random Forest', 'LSTM']

plt.figure(figsize=(8, 5))
plt.bar(model_names, rmse_values, color=['gray', 'orange', 'blue']) # bar plot

plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("RMSE Values")
plt.ylim(0, max(rmse_values) * 1.2)  # Adjust y-axis for better visualization

# Display RMSE values on top of bars
for i, v in enumerate(rmse_values):
    plt.text(i, v + 0.02 * max(rmse_values), f"{v:.2f}", ha='center', fontsize=12)

plt.show()

###
mae_values = [baseline_mae, rf_mae, lstm_mae]
model_names = ['Baseline', 'Random Forest', 'LSTM']

plt.figure(figsize=(8, 5))
plt.bar(model_names, mae_values, color=['gray', 'orange', 'blue']) # bar plot

plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("MAE Values")
plt.ylim(0, max(mae_values) * 1.2)

for i, v in enumerate(mae_values):
    plt.text(i, v + 0.02 * max(mae_values), f"{v:.2f}", ha='center', fontsize=12)

plt.show()

###
r2_values = [baseline_r2, rf_r2, lstm_r2]
model_names = ['Baseline', 'Random Forest', 'LSTM']

plt.figure(figsize=(8, 5))
plt.bar(model_names, r2_values, color=['gray', 'orange', 'blue']) # bar plot

plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("R² Values")
plt.ylim(0, max(r2_values) * 1.2)

for i, v in enumerate(r2_values):
    plt.text(i, v + 0.02 * max(r2_values), f"{v:.2f}", ha='center', fontsize=12)

plt.show()

"""Using Meta-Model because both models capture different patterns"""

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("Random Forest Model initialized and trained.")

min_length = min(len(rf_pred), len(lstm_pred))
rf_pred = rf_pred[:min_length]
lstm_pred = lstm_pred[:min_length]

# Prepare Meta-Model Training Data
X_meta_train = np.column_stack((rf_pred, lstm_pred))  # Stack RF & LSTM Predictions
y_meta_train = y_test[time_steps:]  # Target variable

# Train Linear Regression as Meta-Model
meta_model = LinearRegression()
meta_model.fit(X_meta_train, y_meta_train)

# Generate Final Predictions Using Meta-Model
final_pred_meta = meta_model.predict(X_meta_train)

# Evaluate Meta-Model Ensemble
meta_mse = mean_squared_error(y_meta_train, final_pred_meta)
meta_rmse = meta_mse ** 0.5
print(f"Meta-Model Ensemble RMSE: {meta_rmse:.4f}")

# MAE
meta_mae = mean_absolute_error(y_meta_train, final_pred_meta)
print(f"Meta-Model Ensemble MAE: {meta_mae:.4f}")

# R²
meta_r2 = r2_score(y_meta_train, final_pred_meta)
print(f"Meta-Model Ensemble R²: {meta_r2:.4f}")

"""RMSE Improvement Line Plot"""

# RMSE Improvement Over Baseline
models = ['Baseline', 'Random Forest', 'LSTM', 'Meta-Model Ensemble']
rmse_values = [baseline_rmse, rf_rmse, lstm_rmse, meta_rmse]

plt.figure(figsize=(8, 5))
plt.plot(models, rmse_values, marker='o', linestyle='-', color='dodgerblue', linewidth=2, markersize=8)

plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("Model Performance Improvement (Lower RMSE is Better)")
plt.grid(True)

# Annotate RMSE values
for i, v in enumerate(rmse_values):
    plt.text(i, v + 0.02 * max(rmse_values), f"{v:.2f}", ha='center', fontsize=12)

plt.show()

"""# Error Analysis & Model Explainability

Using SHAP to analyze the RF's feature contributions and interpret its decisions
"""

# Create SHAP Explainer for Random Forest
explainer_rf = shap.Explainer(lambda X: best_rfr.predict(X), X_test)  # Ensuring preprocessing is applied
shap_values_rf = explainer_rf(X_test)  # Compute SHAP values

# Plot SHAP Summary (Feature Importance)
shap.summary_plot(shap_values_rf, X_test, plot_type="bar")  # Feature importance bar chart
shap.summary_plot(shap_values_rf, X_test)  # Beeswarm plot (more detailed)

"""LSTM"""

# Convert LSTM test generator into NumPy array
X_test_lstm_np = np.array([X_test[i] for i in range(len(X_test))])

# Create SHAP Explainer for LSTM
explainer_lstm = shap.Explainer(model_lstm, X_test_lstm_np)
shap_values_lstm = explainer_lstm(X_test_lstm_np)

# Plot LSTM Feature Importance
shap.summary_plot(shap_values_lstm, X_test_lstm_np, plot_type="bar")