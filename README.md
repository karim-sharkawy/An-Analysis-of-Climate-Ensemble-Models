#### Overview: Comparing Random Forest and LSTM Models on West Lafayette Climate Data

Project Plan: West Lafayette, Indiana Weather Prediction Using Machine Learning with Consideration of Climate Change

A coding project which uses machine learning and data from decades of weather patterns of each day to determine yearly climate patterns while incorporating climate change indicators.

#### Goals
1. **Primary Goal**: Develop a machine learning model to predict yearly climate patterns using historical weather data while accounting for climate change effects.
2. **Secondary Goal**: Enhance the model's accuracy and robustness with advanced techniques and external climate data.

I analyzed NOAA’s Local Climatological Data (LCD) for West Lafayette (1973–2024) to predict temperatures using an ensemble of Random Forest Regression (RFR or RF) and Long-Short-Term Memory (LSTM) models.

Building an ensemble model using a combination of Random Forest (RF) and Long Short-Term Memory (LSTM) networks is a powerful and common approach for climate modeling. Both algorithms bring unique strengths that, when combined, can enhance a model’s accuracy and generalizability, giving you the best of both worlds:
- RFRs address feature importance, interpretability, and nonlinearity. They identify and prioritize the most important features in datasets, making it especially useful when there are many variables involved. The regression in RFR means they can handle complex, nonlinear relationships between continuous features. Additionally, RF models are generally easier to interpret than deep learning models, which can be an advantage when analyzing climate-related factors like humidity, wind speed, and pressure (all of which are included in the data).
- On the other hand, LSTM networks excel at capturing temporal dependencies in sequential data. In the context of climate modeling, LSTM can learn and model the relationships between current and past weather conditions, making it ideal for time series data. LSTM networks are particularly strong at remembering long-term dependencies, which is important when dealing with seasonal patterns or long-term trends in climate. By recognizing complex temporal patterns, LSTM can effectively model phenomena like daily or seasonal temperature variations and long-term climate cycles, which are essential for accurate climate predictions.

Specifically, I am be using **Sequential Ensemble (Stacking with Time Series)**. My plan is as follows: first use the RF model to capture the most important features or perform initial preprocessing and feature selection, then pass the output of the RF model as input to the LSTM for time-dependent predictions. This hybrid approach would allow me to combine the RF’s ability to handle features with the LSTM’s ability to capture temporal dependencies.

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
I used Python 3.10.12 on Google Colab. Note that you must change your paths, even when using Google Colab, so they may work on your set up 
1) NumPy 1.26.4
2) Pandas 2.2.0
3) Scikit-learn 1.3.2
4) TensorFlow/Keras 2.15.0
5) Matplotlib 3.7.1
6) Joblib 1.4.2