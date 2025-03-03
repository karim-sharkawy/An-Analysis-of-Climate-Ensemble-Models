"""
ensemble & evaluation
"""

import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load saved models
rfr_model_path = "/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Models/random_forest_climate_model.pkl"
lstm_model_path = "/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Models/lstm_climate_model.h5"

best_rfr = joblib.load(rfr_model_path)
model_lstm = load_model(lstm_model_path)

# Generate Predictions
rf_pred = best_rfr.predict(X_test)
lstm_pred = model_lstm.predict(test_generator).flatten()  # Flatten to match shape

# Ensure correct alignment
print(f"RF Predictions Shape: {rf_pred.shape}, LSTM Predictions Shape: {lstm_pred.shape}")

"""Ensemble Model

Using Meta-Model because both models capture different patterns
"""

# Prepare Meta-Model Training Data
X_meta_train = np.column_stack((rf_pred, lstm_pred))  # Stack RF & LSTM Predictions
y_meta_train = y_test[time_steps:]  # Target variable

# Train Linear Regression as Meta-Model
meta_model = LinearRegression()
meta_model.fit(X_meta_train, y_meta_train)

# Generate Final Predictions Using Meta-Model
final_pred_meta = meta_model.predict(X_meta_train)

# Evaluate Meta-Model Ensemble
meta_rmse = mean_squared_error(y_meta_train, final_pred_meta, squared=False)
print(f"Meta-Model Ensemble RMSE: {meta_rmse:.4f}")

"""Evaluation"""

# Simple Baseline: Predict the mean of training labels
baseline_pred = np.full_like(y_test, y_train.mean())

# Compute RMSE
baseline_rmse = mean_squared_error(y_test, baseline_pred, squared=False)

print(f"Baseline Model RMSE: {baseline_rmse:.4f}")

# RF Model Predictions
rf_pred = best_rfr.predict(X_test)
rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)

# LSTM Model Predictions
lstm_pred = model_lstm.predict(test_generator)
lstm_rmse = mean_squared_error(y_test[time_steps:], lstm_pred, squared=False)

# Print Comparison
print(f"Baseline RMSE: {baseline_rmse:.4f}")
print(f"Random Forest RMSE: {rf_rmse:.4f}")
print(f"LSTM RMSE: {lstm_rmse:.4f}")

"""Visualization improvements of models compared to baseline"""

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