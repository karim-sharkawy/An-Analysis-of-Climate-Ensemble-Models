"""
**Data collection**
These notebooks are designed to be read with my [code documentation](Comparison of Random Forest and LSTM models on West Lafayette Climate Data.pdf) in mind!
"""

import glob
import os
import pandas as pd
import numpy as np

# I used google colab, feel free to use whatever environment
# Make sure to remove google.colab and drive mounts and change paths

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

"""# Visualizations

For the most important features
"""

hourlydew = df_klima['HourlyDewPointTemperature']
hourlydew.plot()

hourlydry = df_klima['HourlyDryBulbTemperature']
hourlydry.plot()