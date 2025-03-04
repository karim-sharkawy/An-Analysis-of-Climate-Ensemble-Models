### prediction_model


import joblib
from keras.models import load_model

# load models and make combined predictions
def load_models():
    pipeline_rf = joblib.load('random_forest_climate_model.pkl')
    model_lstm = load_model('lstm_climate_model.h5')
    return pipeline_rf, model_lstm

def make_predictions(X_new, time_steps=10):
    pipeline_rf, model_lstm = load_models()

    # Preprocess new data
    X_new_scaled = pipeline_rf.named_steps['preprocessor'].transform(X_new)

    # Make predictions with Random Forest
    y_pred_rf = pipeline_rf.predict(X_new)

    # Prepare data for LSTM predictions
    new_generator = TimeseriesGenerator(X_new_scaled, np.zeros(len(X_new_scaled)), length=time_steps, batch_size=1)
    y_pred_lstm = model_lstm.predict(new_generator)
    y_pred_lstm = y_pred_lstm.flatten()

    # Combine predictions (simple average)
    combined_predictions = (y_pred_rf[time_steps:] + y_pred_lstm) / 2

    return combined_predictions