# RFR model

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

from google.colab import drive
drive.mount('/content/drive')

"""# Feature Importance"""

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