### Ensemble Climate Model full code

import glob
import os
import pandas as pd
import numpy as np
import joblib # saving RFR model

from sklearn.model_selection import train_test_split  # training and testing sets
from sklearn.model_selection import RandomizedSearchCV  # efficient hyperparameter tuning with randomized search
from sklearn.model_selection import GridSearchCV  # hyperparameter tuning in a refined range
from sklearn.preprocessing import StandardScaler  # standardize features
from sklearn.pipeline import Pipeline  # combines preprocessing steps and model training into one workflow
from sklearn.impute import SimpleImputer  # handle missing data
from sklearn.ensemble import RandomForestRegressor  # RFR
from sklearn.linear_model import LinearRegression # for meta model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # evaluating model performance
from sklearn.feature_selection import SelectFromModel  # selecting important features

!pip install keras-tuner
from tensorflow.keras.models import load_model  # loading a saved Keras model (LSTM in this case)
from tensorflow.keras.models import Sequential  # sequential (layer-stacked) neural network
from tensorflow.keras.layers import LSTM, Dropout, Dense # LSTM for sequential learning, Dense for output layers
from tensorflow.keras.callbacks import EarlyStopping # preventing overfitting and overusing resources
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # Creates time-series data batches for LSTM
from kerastuner.tuners import BayesianOptimization # hyperparamter tuning
from sklearn.metrics import mean_squared_error, r2_score # evaluating model performance

import shap # explainability
import matplotlib.pyplot as plt # visualizations

from google.colab import drive
drive.mount('/content/drive')

"""# Combining CSV Files & Loading DF"""

data_path = '/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Data'

csv_files = glob.glob(os.path.join(data_path, '*.csv')) # find all CSV files in folder
print(csv_files)

data_frames = []
for file in csv_files:
    df = pd.read_csv(file)
    data_frames.append(df)

df_klima_original = pd.concat(data_frames, ignore_index=True)
df_klima_original.to_csv('/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Data/unclean_climate_data.csv', index=False) # save to new CSV

df_klima_original.head(n=2)

"""Load & copy DF, work with the copy DF"""

data_path = '/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Data'

df_klima_original = pd.read_csv(os.path.join(data_path, 'unclean_climate_data.csv'))
df_klima = df_klima_original.copy()

# remove apostrophes to see the full output
'''
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
'''

df_klima

df_klima_original.head(n=2)

"""# Cleaning

Most columns had a large number of null/non-int values, so I set a 'threshhold' at 'HourlySeaLevelPressure'. After that, every column had a small number of values compared to the number of rows
"""

non_null_counts = df_klima.notnull().sum()

summary_df = pd.DataFrame({
    'Column': non_null_counts.index,
    'Non-Null Count': non_null_counts.values
}) # creating summary DF

# Sort the summary DF by Non-Null Count in descending order
sorted_summary_df = summary_df.sort_values(by='Non-Null Count', ascending=False)
print(sorted_summary_df) # we'll manually find the cutoff here

"""After searching through the `summary_df`, I decided to keep anything with a non-null count higher than `HourlySeaLevelPressure`"""

threshold = non_null_counts['HourlySeaLevelPressure']
columns_to_keep = non_null_counts[non_null_counts >= threshold].index

# removing columns that don't have meaning
columns_to_remove = ['STATION', 'REPORT_TYPE', 'REPORT_TYPE.1', 'SOURCE', 'SOURCE.1', 'REM', 'WindEquipmentChangeDate']
columns_to_remove = [col for col in columns_to_remove if col in columns_to_keep]

# create a new DF with columns_to_keep but not columns_to_remove
df_klima = df_klima[columns_to_keep]
df_klima = df_klima.drop(columns=columns_to_remove)

"""**Fixing text/missing values**

There are some parts of the data where a character is given with or instead of the number value provided. These characters are:

1. "T" = trace precipitation amount or snow depth (an amount too small to measure, usually < 0.005 inches water equivalent) (appears instead of numeric value)
2. "s" = suspect value (appears with value)
3. "M" = missing value (appears instead of value)
4. Blank = value is unreported (nothing appears or "NaN")
5. "*" = Amounts included in following measurement; time distribution unknown
- for temps, these are used to indicate the extreme for the day and month, these can be deleted

Below shows how we dealt with them

"""

exclude_columns = ['DATE'] # Columns to exclude from filling missing values

# Define the function to clean the data
def clean_data(value):
    if value == 'T':
        return 0.0025  # A small value representing trace amount
    elif value == 's':
        return np.nan  # Suspect value can be treated as missing value
    elif value == 'M':
        return np.nan  # Missing value
    elif value == '*' or value == '':
        return np.nan  # Unreported value
    else:
        try:
            # Convert to float if possible
            return float(value)
        except ValueError:
            return np.nan  # If the value cannot be converted to float, treat it as missing

# Apply the cleaning function to the entire DataFrame except for the excluded columns
for col in df_klima.columns:
    if col not in exclude_columns:
        df_klima[col] = df_klima[col].apply(clean_data)

# Define a function to compute window mean
def compute_window_mean(df, col, idx, window_size):
    # Define the window
    start_idx = max(0, idx - window_size)
    end_idx = min(len(df), idx + window_size + 1)

    # Extract the window of values
    window_values = df[col].iloc[start_idx:end_idx]

    # Drop NaN values from the window and calculate mean
    non_nan_values = window_values.dropna()
    if len(non_nan_values) > 0:
        return non_nan_values.mean()
    return np.nan

# Define a function to fill missing values
def fill_missing_values(df, exclude_cols, window_size=10):
    # First pass: Fill missing values with mean of full windows
    df_filled = df.copy()
    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            for idx in df[df[col].isna()].index:
                mean_value = compute_window_mean(df, col, idx, window_size)
                if not pd.isna(mean_value):
                    df_filled.at[idx, col] = mean_value

    # Second pass: Handle any remaining missing values
    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            for idx in df_filled[df_filled[col].isna()].index:
                mean_value = compute_window_mean(df, col, idx, window_size)
                if pd.isna(mean_value):
                    # If we cannot calculate a mean, use the column's global mean
                    mean_value = df[col].mean()
                df_filled.at[idx, col] = mean_value

    return df_filled

# Apply the function
df_klima = fill_missing_values(df_klima, exclude_columns)

"""**Identifying and managing outliers**"""

def impute_outliers(data, strategy='median', threshold=1.5):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (threshold * IQR)
    upper_bound = Q3 + (threshold * IQR)

    if strategy == 'median':
        fill_value = np.median(data)
    elif strategy == 'mean':
        fill_value = np.mean(data)
    else:
        raise ValueError("Strategy must be 'median' or 'mean'")

    data = np.where((data < lower_bound) | (data > upper_bound), fill_value, data)
    return data

for i in df_klima.columns:
  if i not in exclude_columns:
    df_klima[i] = impute_outliers(df_klima[i])

"""**Setting the date as the index**"""

df_klima['DATE'] = pd.to_datetime(df_klima['DATE'], format='ISO8601')
df_klima.set_index('DATE', inplace=True)
df_klima.sort_index(inplace=True) # Sort DF by date

df_klima.head(n=3)

df_klima.describe()

"""Finally, we have clean data that we can now work with :)"""

df_klima.to_csv('/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Data/climate_data.csv', index=True)

"""# Feature Engineering"""

csv_path = '/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Data/climate_data.csv'
df_klima = pd.read_csv(csv_path)

# Need to make the date the index everytime :)
df_klima['DATE'] = pd.to_datetime(df_klima['DATE'], format='ISO8601')
df_klima.set_index('DATE', inplace=True)
df_klima.sort_index(inplace=True) # Sort DF by date

"""Removing highly correlated features to reduce noise and preventing multicollinearity"""

corr_matrix = df_klima.corr(method='pearson', numeric_only=True)

ceiling = 0.7999  # upper bound to reduce redundancy
floor = 0.2999  # lower bound to drop weak predictors
columns_to_drop = set()  # using a set to avoid duplicates

# removing low-correlation features
for column in df_klima.columns:
    if abs(corr_matrix[column].drop(column).max()) < floor:
        columns_to_drop.add(column)

# removing highly correlated pairs
checked_pairs = set()
for column in df_klima.columns:
    for correlated_column in df_klima.columns:
        if column != correlated_column and (column, correlated_column) not in checked_pairs:
            if abs(corr_matrix.loc[column, correlated_column]) > ceiling:
                target_corr_column = abs(corr_matrix[column].drop(column)).max() # drop feature with lower correlation to target
                target_corr_correlated = abs(corr_matrix[correlated_column].drop(correlated_column)).max()

                if target_corr_column > target_corr_correlated:
                    columns_to_drop.add(correlated_column)
                else:
                    columns_to_drop.add(column)

                # marking checked pais
                checked_pairs.add((column, correlated_column))
                checked_pairs.add((correlated_column, column))

df_klima = df_klima.drop(columns=columns_to_drop)
print(f'Dropped columns: {columns_to_drop}')

# Define new file path for preprocessed data
preprocessed_csv_path = os.path.splitext(csv_path)[0] + "_preprocessed.csv"

# Save the cleaned dataset
df_klima.to_csv(preprocessed_csv_path)

print(f"Preprocessed dataset saved to: {preprocessed_csv_path}")

df_klima.head()

# Define input file path
csv_path = '/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Data/climate_data_preprocessed.csv'

# Load dataset
df_klima = pd.read_csv(csv_path)
df_klima['DATE'] = pd.to_datetime(df_klima['DATE'], format='ISO8601')
df_klima.set_index('DATE', inplace=True)
df_klima.sort_index(inplace=True)

"""Feature importance on a temporary RFR"""

X = df_klima.drop(columns=['HourlyDryBulbTemperature'])
y = df_klima['HourlyDryBulbTemperature']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Train a temporary RFR to get feature importance
temp_rfr = RandomForestRegressor(n_estimators=100, random_state=42)
temp_rfr.fit(X_train, y_train)

# Get feature importance scores
feature_importances = temp_rfr.feature_importances_

# Select top N features (you can tweak the threshold)
feature_names = X_train.columns  # Store original feature names
important_features = np.argsort(feature_importances)[-3:]  # Selecting top 3 features

# Extract the names of the top features
top_feature_names = feature_names[important_features]  # Use the indices to get the column names
print(f"Selected top features: {top_feature_names}")

# Reduce dataset to only these features
X_train_selected = X_train[top_feature_names]  # Use column names to select features
X_test_selected = X_test[top_feature_names]

# Now our train and test datasets are refined further using feature importance!
X_train = X_train_selected
X_test = X_test_selected

"""# RFR

Finding the best Random Forest Regression (RFR) Model using RandomizedSearchCV
"""

param_grid = {
    'model__n_estimators': [50, 100, 200, 250],  # number of trees
    'model__max_depth': [None, 10, 20, 30],  # tree depth
    'model__min_samples_split': [2, 5, 10],  # min samples required to split a node
    'model__min_samples_leaf': [1, 2, 4],  # min samples per leaf
    'model__max_features': ['auto', 'sqrt', 'log2']  # number of features per split
} # quicker than GridSearchCV

pipeline_rfr = Pipeline([
    ('preprocessor', SimpleImputer(strategy='mean')), # already preprocessed, but just for consistency
    ('model', RandomForestRegressor(random_state=42))
])

random_search = RandomizedSearchCV(
    pipeline_rfr,
    param_distributions=param_grid,
    n_iter=4,  # Number of random combinations to try
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1  # Use all available CPU cores
) # hyperparameter tuning

random_search.fit(X_train, y_train) # training
best_rfr = random_search.best_estimator_ # Get the best model

# Evaluating the best model
y_pred = best_rfr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {random_search.best_params_}")
print(f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, R Squared Score: {r2:.4f}")

"""Using GridSearchCV to refine model performance based on RandomizedSearchCV results"""

keys = list(random_search.keys())

param_grid = {
    'model__n_estimators': [random_search[keys[0]]*0.8, random_search[keys[0]], random_search[keys[0]]*1.2],
    'model__max_depth': [random_search[keys[4]]*0.8, random_search[keys[4]], random_search[keys[4]]*1.2],  # tree depth
    'model__min_samples_split': [random_search[keys[1]] - 1, random_search[keys[1]], random_search[keys[1]]*2],  #  node splitting
    'model__min_samples_leaf': [random_search[keys[2]] - 0.5, random_search[keys[2]], random_search[keys[2]]*2]
} # refined hyperparameter grid

pipeline_rfr = Pipeline([
    ('preprocessor', Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])),
    ('model', RandomForestRegressor(random_state=42))
]) # consistency

grid_search = GridSearchCV(
    estimator=pipeline_rfr,
    param_grid=param_grid,
    cv=5, # ensures the model generalizes well
    scoring='neg_mean_squared_error',  # find the lowest MSE model
    verbose=2,
    n_jobs=-1  # Use all available cores for efficiency, check default
)

grid_search.fit(X_train, y_train)

best_rfr = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

rfr_model_path = '/content/drive/My Drive/Colab Notebooks/Ensemble Climate Model/Models/random_forest_climate_model.pkl'
joblib.dump(best_rfr, rfr_model_path) # saving the model

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