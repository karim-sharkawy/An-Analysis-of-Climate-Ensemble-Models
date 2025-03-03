### Notes for each file (in order)

#### 1) data_cleaning.py 
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
- Replaced specific character codes (`'T'` for trace precipitation, `'M'` for missing values) with meaningful values or NaN (shown below).

```python
def clean_data(value):
    if value == 'T': return 0.0025  # Trace value
    elif value in ['s', 'M', '*', '']: return np.nan
    return float(value) if isinstance(value, str) and value.isnumeric() else np.nan
```

### 2) rfr_model.py
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



### 3) lstm_model.py
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



### 4) ensemble_&_evaluation.py








To compare both models, I visualized:  
- MSE loss: Random Forest outperformed LSTM with significantly lower errors.
- Computation time: LSTM was slower due to its sequential architecture.


### 5) prediction_model.py

### Room for improvement:
1. Parallelism and Distributed Training for Ensemble Models to handle intensive computation. Hence, we can:
- RFs is inherently parallelizable because each tree in the forest is built independently. Use scikit-learn and XGBoost to take advantage of multi-core processors and train multiple trees in parallel.
- For LSTMs, TensorFlow supports distributed training across multiple GPUs or machines, processing larger data in less time.

2. Efficient Ensemble Methods
- Ensemble models are quite large, we can use XGBoost and bagging methods to help with this

3. Mini-batches for LSTMS
- Instead of feeding the entire dataset into the model at once, break it down into batches. This allows the model to learn in smaller, more manageable chunks, reducing the memory load and making it easier to fit large datasets into memory.

4. Temporal & Spatial Analysis: Detect seasonal patterns and long-term climate trends.  
   - Apply moving averages and Fourier transforms to extract seasonal effects.

5. Advanced Enhancements: Make the project more dynamic and applicable to real-world scenarios.  
1. Automated Data Pipeline  
   - Develop a pipeline to continuously update the model with new data from NOAA.  
2. Incorporate Climate Change Factors  
   - Integrate climate change data (CO₂ levels, temperature anomalies) into the model.  
   - Ensure seamless integration with existing weather data.