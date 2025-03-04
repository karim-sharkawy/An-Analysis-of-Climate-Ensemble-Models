# Description of each script (in order)

### 1) data_cleaning.py 
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
The code sets up a machine learning pipeline to train and evaluate a Random Forest Regression (RFR) model using climate data. The data is first loaded from a CSV file stored on Google Drive, with the 'DATE' column processed as a datetime object and set as the index. Features that are highly correlated with others are removed to avoid multicollinearity, which can negatively affect the model’s performance. After preprocessing, the data is split into training and testing sets, and a pipeline is created that includes imputation of missing values and scaling of features. The hyperparameters of the Random Forest model are fine-tuned using `RandomizedSearchCV`, a more efficient method than `GridSearchCV`, which explores different combinations of model parameters through cross-validation.

Once the model is trained, its performance is evaluated using metrics such as Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R² score. The process is followed by further refinement of the model's hyperparameters using `GridSearchCV`, which narrows the search space to optimize the model’s performance further. The final model is saved to disk using `joblib` for future predictions. The code includes steps to ensure compatibility with Google Colab environments, such as mounting Google Drive and specifying the correct file paths. Additionally, careful handling of the feature selection process, hyperparameter tuning, and model evaluation ensures the model is both robust and efficient.

```
# Creating the pipeline for preprocessing and model training
pipeline_rfr = Pipeline([
    ('preprocessor', SimpleImputer(strategy='mean')),  # Handling missing values
    ('model', RandomForestRegressor(random_state=42))  # Random Forest model
])

# RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    pipeline_rfr,
    param_distributions=param_grid,
    n_iter=4,  # Number of random combinations to try
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1  # Use all available CPU cores
)

# Fit the model
random_search.fit(X_train, y_train)

# Save the best model
joblib.dump(random_search.best_estimator_, rfr_model_path)
```


### 3) lstm_model.py
The code builds and trains an LSTM (Long Short-Term Memory) model for climate prediction using a dataset loaded from Google Drive. The dataset is preprocessed, including date handling, feature selection with a pre-trained Random Forest model, and scaling of selected features. The top features identified by the Random Forest model are used as inputs to the LSTM, and additional feature engineering steps like creating moving averages and daily averages are applied to smooth the data and reduce noise. After feature engineering, the data is split into training and testing sets, and time-series data generators are used to prepare the data for the LSTM model. This step is crucial for handling the sequential nature of the data and feeding it into the LSTM network.

The LSTM model is built using Keras, with two LSTM layers and dropout layers to prevent overfitting. The model is trained on the prepared time-series data using early stopping to halt training if validation loss does not improve for a set number of epochs. For hyperparameter tuning, the code uses `kerastuner`'s Bayesian Optimization to optimize model parameters like the number of units in the LSTM layers, dropout rates, and learning rates. Once the best model is found, it is evaluated on the test data, and the root mean square error (RMSE) is calculated to assess model performance. Finally, the trained model is saved to disk for future use.

```
# Building the LSTM model with dropout layers and tuning hyperparameters
model_lstm = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(time_steps, X_train_scaled.shape[1])),
    Dropout(0.2),
    LSTM(30, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output layer for regression
])

# Compile the model
model_lstm.compile(optimizer='adam', loss='mse')

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the LSTM model with time-series generators
history = model_lstm.fit(
    train_generator,
    epochs=50,
    validation_data=test_generator,
    callbacks=[early_stop]
)

# Hyperparameter tuning with Bayesian Optimization
tuner = BayesianOptimization(
    build_model,
    objective='val_loss',  # Minimize validation loss
    max_trials=10,  # Number of trials
    executions_per_trial=3,  # Number of executions for each trial
    directory='tuner_output',
    project_name='lstm_rf_tuning'
)

tuner.search(train_generator, epochs=50, validation_data=test_generator, callbacks=[early_stop])
best_model = tuner.get_best_models()[0]
best_model.evaluate(test_generator)  # Evaluate the best model
```


### 4) ensemble_&_evaluation.py

This script builds an ensemble model by combining predictions from the RF and LSTM models using weighted averaging and meta-model stacking (linear regression). It evaluates model performance by comparing RMSE scores against a baseline model and determining which approach yields the most accurate predictions. The script leverages `joblib` to load the trained RF model and `tensorflow.keras.models` to load the LSTM model, ensuring seamless integration of both components. Additionally, `sklearn.metrics` is used to compute RMSE, helping quantify improvements over baseline predictions.  

To enhance interpretability, the script includes multiple visualizations using `matplotlib` and `shap`. A feature importance comparison between RF’s built-in rankings and SHAP values provides insights into how each model makes decisions. A line plot of RMSE improvements highlights the effectiveness of different models, while actual vs. predicted scatter plots reveal potential biases or inconsistencies. These evaluations and visualizations ensure a comprehensive understanding of model performance before final deployment.


### 5) prediction_model.py
In progress.

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