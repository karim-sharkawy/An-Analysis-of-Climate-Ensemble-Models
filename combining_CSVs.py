import glob
import os
import pandas as pd
from google.colab import drive # I used colab, feel free to use something else

drive.mount('/content/drive')
data_path = '/content/drive/...'

csv_files = glob.glob(os.path.join(data_path, '*.csv')) # find all CSVs in your folder
print(csv_files)

data_frames = []
for file in csv_files:
    df = pd.read_csv(file)
    data_frames.append(df)

climate_df = pd.concat(data_frames, ignore_index=True)
climate_df.to_csv('WLClimateData.csv', index=False) # save to new CSV

pd.set_option('display.max_columns', None)
climate_df # check when the data starts and ends