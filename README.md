# An Ensemble of Random Forest and LSTM Models on Climate Data with Consideration of Climate Change

A coding project which uses machine learning and hourly data from NOAA’s Local Climatological Data (LCD) for West Lafayette (1973–2024) to predict temperatures using an ensemble model of Random Forest Regression (RFR or RF) and Long-Short-Term Memory (LSTM) models while incorporating climate change indicators. My primary goal was to develop a machine learning model to predict yearly climate patterns using historical weather data while accounting for climate change effects. My secondary goal was then to enhance that model's accuracy with external climate data. 

General project details are on this README, along with the dependencies. For further explanation on results, discussion of results, and data cleaning and feature engineering, please go to the READMEs of each folder where more information is provided.

Building an ensemble model using a combination of Random Forest (RF) and Long Short-Term Memory (LSTM) networks is a powerful and common approach in the climate modeling community. Both algorithms bring unique strengths that, when combined, can enhance a model’s accuracy and generalizability, giving you the best of both worlds:
- RFRs address feature importance, interpretability, and nonlinearity. They identify and prioritize the most important features in datasets, making it especially useful when there are many variables involved. The regression in RFR means they can handle complex, nonlinear relationships between continuous features. Additionally, RF models are generally easier to interpret than deep learning models, which can be an advantage when analyzing climate-related factors like humidity, wind speed, and pressure (all of which are included in the data).
- On the other hand, LSTM networks excel at capturing temporal dependencies in sequential data. In the context of climate modeling, LSTM can learn and model the relationships between current and past weather conditions, making it ideal for time series data. LSTM networks are particularly strong at remembering long-term dependencies, which is important when dealing with seasonal patterns or long-term trends in climate. By recognizing complex temporal patterns, LSTM can effectively model phenomena like daily or seasonal temperature variations and long-term climate cycles, which are essential for accurate climate predictions.

Specifically, I am be using **Sequential Ensemble (Stacking with Time Series)**. My plan is as follows: first use the RF model to capture the most important features or perform initial preprocessing and feature selection, then pass the output of the RF model as input to the LSTM for time-dependent predictions. This hybrid approach would allow me to combine the RF’s ability to handle features with the LSTM’s ability to capture temporal dependencies.

# Dependencies
I used Python 3.10.12 on Google Colab with the following libraries:
1) NumPy 1.26.4
2) Pandas 2.2.0
3) Scikit-learn 1.3.2
4) TensorFlow/Keras 2.15.0
5) Matplotlib 3.7.1
6) Joblib 1.4.2
7) 0.46.0

# Results
This is dicussed much further in the "Results & Visualizations" folder, but here is a summary. The LSTM model outperforms all other models, achieving the lowest RMSE and MAE, along with an R² of 0.99, indicating exceptional predictive accuracy. This model significantly improves over the baseline, reducing RMSE by 90.39% and MAE by 92.35%. The Random Forest model also shows improvement over the baseline, with a 12.31% reduction in RMSE and 15.58% reduction in MAE, but its lower R² of 0.23 suggests potential underfitting. The Meta-Model Ensemble, combining Random Forest and LSTM predictions, failed to improve upon the baseline, indicating that the ensemble approach did not leverage the individual model strengths effectively. Overall, the LSTM model stands out as the most effective, while the Random Forest and ensemble models require further tuning or more advanced techniques to improve performance.
