#### Overview: Comparing Random Forest and LSTM Models on West Lafayette Climate Data  

I analyzed NOAA’s Local Climatological Data (LCD) for West Lafayette (2020–2024) to predict hourly wet bulb temperatures using Random Forest and LSTM models.

### Steps and Code Highlights  

#### 1. Data Preparation 
The dataset required careful preprocessing to handle missing and non-numeric values. I:
- Dropped irrelevant columns based on a threshold of non-null values.
- Filled missing values using a *window-based mean* approach to preserve temporal relationships in the time series. This is detailed in the comments.
```
def compute_window_mean(df, col, idx, window_size):
    start_idx = max(0, idx - window_size)
    end_idx = min(len(df), idx + window_size + 1)
    
    window_values = df[col].iloc[start_idx:end_idx].dropna()
    return window_values.mean() if not window_values.empty else np.nan

def fill_missing_values(df, exclude_cols, window_size=10):
    df_filled = df.copy()
    
    for col in df.columns:
        if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
            for idx in df[df[col].isna()].index:
                mean_value = compute_window_mean(df, col, idx, window_size)
                if pd.isna(mean_value):
                    mean_value = df[col].mean()  # Fallback to column mean
                df_filled.at[idx, col] = mean_value

    return df_filled
```
- 
- Replaced specific character codes (`'T'` for trace precipitation, `'M'` for missing values) with meaningful values or NaN (shown below).

```python
def clean_data(value):
    if value == 'T': return 0.0025  # Trace value
    elif value in ['s', 'M', '*', '']: return np.nan
    return float(value) if isinstance(value, str) and value.isnumeric() else np.nan
```

#### 2. Random Forest Model
I chose Random Forest for its ability to handle tabular data efficiently. Using 10-fold cross-validation:
- I tuned hyperparameters like `n_estimators` (trees) and `max_depth` for balance between underfitting and overfitting.
- Tracked computation time and loss (MSE) per fold:

```python
kf = KFold(n_splits=10, shuffle=True, random_state=42)
for train_idx, val_idx in kf.split(X_train):
    pipeline_rf.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
    y_pred = pipeline_rf.predict(X_train.iloc[val_idx])
    mse = mean_squared_error(y_train.iloc[val_idx], y_pred)
```

#### 3. LSTM Model
I implemented an LSTM network to capture temporal dependencies in the weather data:
- Used a time-series generator to prepare sequential input-output pairs.
- Set up an LSTM architecture with 50 hidden units and trained it over 10 epochs:

```python
model_lstm = Sequential([
    LSTM(50, activation='relu', input_shape=(time_steps, X_train.shape[1])),
    Dense(1)
])
model_lstm.compile(optimizer='adam', loss='mse')
model_lstm.fit(train_generator, epochs=10, validation_data=test_generator)
```

#### 4. Model Comparison
To compare both models, I visualized:  
- MSE loss: Random Forest outperformed LSTM with significantly lower errors.
- Computation time: LSTM was slower due to its sequential architecture.

### Results  
- Random Forest excelled with lower MSE and faster training times, making it ideal for small datasets like this.
- LSTM struggled to leverage its temporal advantages, likely due to the dataset’s limited variability and size.

# Dependencies
1) Python 3.10.12
2) NumPy 1.26.4
3) Pandas 2.2.0
4) Scikit-learn 1.3.2
5) TensorFlow 2.15.0
6) Keras 2.15.0
7) Matplotlib 3.7.1
