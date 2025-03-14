### LSTM model

import os
import pandas as pd
import numpy as np
import joblib # Saving RF model
import matplotlib.pyplot as plt  # Plot feature importance

!pip install keras-tuner
from tensorflow.keras.models import load_model  # Loads a saved Keras model
from tensorflow.keras.models import Sequential  # Defines a sequential (layer-stacked) neural network
from tensorflow.keras.layers import LSTM, Dropout, Dense # LSTM for sequential learning, Dense for output layers
from tensorflow.keras.callbacks import EarlyStopping # preventing overfitting and overusing resources
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # Creates time-series data batches for LSTM
from kerastuner.tuners import BayesianOptimization # hyperparamter tuning
from sklearn.metrics import mean_squared_error, r2_score # evaluating model performance

from sklearn.model_selection import train_test_split  # Splits data into training and testing sets
from sklearn.preprocessing import StandardScaler  # Standardizes features to improve model performance
from sklearn.pipeline import Pipeline  # Combines preprocessing steps and model training into one workflow
from sklearn.metrics import mean_squared_error # evaluating model performance
from sklearn.feature_selection import SelectFromModel  # Selects important features based on model output

from google.colab import drive # Feel free to use whatever environment
drive.mount('/content/drive') # Make sure to remove google.colab and drive mounts and change paths

csv_path = '/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Data/climate_data_preprocessed.csv'
df_klima = pd.read_csv(csv_path)

# Need to make the date the index everytime :)
df_klima['DATE'] = pd.to_datetime(df_klima['DATE'], format='ISO8601')
df_klima.set_index('DATE', inplace=True)
df_klima.sort_index(inplace=True) # Sort DF by date

df_klima.head(3)

"""# LSTM"""

rfr_model_path = "/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Models/random_forest_climate_model.pkl"
best_rfr = joblib.load(rfr_model_path)

"""Passing selected features into LSTM model"""

# Ensure target variable is also aggregated the same way
y_lstm = df_klima['HourlyDryBulbTemperature']

# Train-test split after feature engineering
train_size = int(0.8 * len(df_klima))
X_train_lstm, X_test_lstm = df_klima.iloc[:train_size], df_klima.iloc[train_size:]
y_train_lstm, y_test_lstm = y_lstm.iloc[:train_size], y_lstm.iloc[train_size:]


# Standardize only after feature selection & aggregation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_lstm)
X_test_scaled = scaler.transform(X_test_lstm)

# Time-series data generation for LSTM
time_steps = 10
batch_size = 32 # look into tuning this

train_generator = TimeseriesGenerator(X_train_scaled, y_train_lstm, length=time_steps, batch_size=batch_size)
test_generator = TimeseriesGenerator(X_test_scaled, y_test_lstm, length=time_steps, batch_size=batch_size)

print(f"y_pred shape: {y_train_lstm.shape}, y_test shape: {y_test_lstm.shape}")  # Ensure alignment

# Build the LSTM model (same as before)
model_lstm = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, X_train_scaled.shape[1])),
    Dropout(0.2),
    LSTM(30, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output layer for regression
])

model_lstm.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Prevent overfitting

# Train LSTM
history = model_lstm.fit(
    train_generator,
    epochs=15,
    validation_data=test_generator,
    callbacks=[early_stop]
)

"""Hyperparameter Tuning"""

def build_model(hp):
    model = Sequential()

    model.add(LSTM(units=hp.Int('units_1', min_value=50, max_value=200, step=50),
                   activation='relu',
                   return_sequences=True,
                   input_shape=(time_steps, X_train_scaled.shape[1])))  # Ensure correct feature count
    model.add(Dropout(hp.Float('dropout_1', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(LSTM(units=hp.Int('units_2', min_value=30, max_value=100, step=30),
                   activation='relu'))
    model.add(Dropout(hp.Float('dropout_2', min_value=0.1, max_value=0.5, step=0.1)))

    model.add(Dense(1))  # Output layer

    model.compile(optimizer='adam', loss='mse')

    return model

# Initialize the BayesianOptimization tuner
tuner = BayesianOptimization(
    build_model,
    objective='val_loss',  # Objective to minimize
    max_trials=3,  # Number of hyperparameter combinations to try
    executions_per_trial=3,  # Number of executions for each trial
    directory='tuner_output',
    project_name='lstm_rf_tuning'
) # experiment with these within your context!

early_stop = EarlyStopping(monitor='val_loss', patience=5) # to prevent overfitting

tuner.search(
    train_generator,
    epochs=15, # experiment with this!
    validation_data=test_generator,
    callbacks=[early_stop]
)

best_model = tuner.get_best_models()[0] # Get the best model
best_model.evaluate(test_generator) # Evaluate the best model on the test data

"""Evaluation & Saving"""

y_pred = best_model.predict(test_generator)

# Get the correct length of y_test for comparison
y_test_actual = np.array(y_test_lstm[len(y_test_lstm) - len(y_pred):])  # Match lengths

# Compute RMSE
mse = mean_squared_error(y_test_actual, y_pred)
rmse = mse ** 0.5
print(f"Test RMSE: {rmse:.4f}")

lstm_model_path = '/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Models/lstm_climate_model.h5'
model_lstm.save(lstm_model_path) # saving the model