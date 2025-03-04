### ensemble & evaluation

import joblib
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import shap # explainability
import matplotlib.pyplot as plt # visualizations

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

"""# Ensemble Model

Using weighted averages if models perform the same
"""

# Assign Weights Based on Performance (Lower RMSE = Higher Weight)
rf_rmse = mean_squared_error(y_test, rf_pred, squared=False)
lstm_rmse = mean_squared_error(y_test[time_steps:], lstm_pred, squared=False)

rf_weight = 1 / rf_rmse
lstm_weight = 1 / lstm_rmse

# Normalize Weights
rf_weight /= (rf_weight + lstm_weight)
lstm_weight /= (rf_weight + lstm_weight)

# Compute Final Predictions
final_pred_weighted = (rf_weight * rf_pred) + (lstm_weight * lstm_pred)

# Evaluate Weighted Ensemble
ensemble_rmse = mean_squared_error(y_test[time_steps:], final_pred_weighted, squared=False)
print(f"Weighted Ensemble RMSE: {ensemble_rmse:.4f}")

"""Using Meta-Model because both models capture different patterns"""

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

"""# Visualization

Feature Importance Comparison (RF vs. SHAP)
"""

# Compare RF's built-in feature importance with SHAP
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Random Forest Feature Importance (Traditional)
ax[0].barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
ax[0].set_title("Random Forest Feature Importance (Built-in)")
ax[0].set_xlabel("Importance Score")
ax[0].invert_yaxis()

# SHAP Feature Importance
shap_values_rf = explainer_rf(X_test)
shap_importance = np.abs(shap_values_rf.values).mean(axis=0)
shap_feature_importance = pd.DataFrame({'Feature': X_test.columns, 'Importance': shap_importance}).sort_values(by="Importance", ascending=False)

ax[1].barh(shap_feature_importance['Feature'], shap_feature_importance['Importance'], color='lightcoral')
ax[1].set_title("SHAP Feature Importance")
ax[1].set_xlabel("SHAP Value Mean Abs")

plt.tight_layout()
plt.show()

"""RMSE Improvement Line Plot"""

# RMSE Improvement Over Baseline
models = ['Baseline', 'Random Forest', 'LSTM', 'Weighted Ensemble', 'Meta-Model Ensemble']
rmse_values = [baseline_rmse, rf_rmse, lstm_rmse, ensemble_rmse, meta_rmse]

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

"""Actual vs. Predicted Scatter Plot (RF & LSTM)"""

# Plot Actual vs. Predicted for LSTM & RF
fig, ax = plt.subplots(1, 2, figsize=(15, 5))

# Random Forest Predictions
ax[0].scatter(y_test, rf_pred, color='orange', alpha=0.5)
ax[0].plot(y_test, y_test, color='black', linestyle='dashed')  # Perfect predictions line
ax[0].set_title("Random Forest: Actual vs. Predicted")
ax[0].set_xlabel("Actual Values")
ax[0].set_ylabel("Predicted Values")

# LSTM Predictions
ax[1].scatter(y_test[time_steps:], lstm_pred, color='blue', alpha=0.5)
ax[1].plot(y_test[time_steps:], y_test[time_steps:], color='black', linestyle='dashed')
ax[1].set_title("LSTM: Actual vs. Predicted")
ax[1].set_xlabel("Actual Values")
ax[1].set_ylabel("Predicted Values")

plt.tight_layout()
plt.show()

"""# Error Analysis & Model Explainability

Using SHAP to analyze the RF's feature contributions and interpret its decisions
"""

# Create SHAP Explainer for Random Forest
explainer_rf = shap.Explainer(best_rfr.named_steps['model'], X_test)  # Explain RF model predictions
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