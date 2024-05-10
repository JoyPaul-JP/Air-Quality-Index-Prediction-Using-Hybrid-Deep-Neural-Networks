import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.models import load_model
from joblib import load
from keras import backend as K

# Load the saved model
loaded_ann_model = load_model('F:/Works/Application of ML/Final/ANN_trained_model.h5')

# Compile the loaded ANN model with the custom optimizer
loaded_ann_model.compile(optimizer=Adam(learning_rate=0.0001 ), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

# Load the saved model with custom objects
loaded_lstm_model = load_model('F:/Works/Application of ML/Final/LSTM_trained_model.h5', custom_objects={'mse': mse})

# Load the scaler for LSTM
scaler_LSTM = load('F:/Works/Application of ML/Final/scaler_LSTM.joblib')

# Load the scaler for ANN
scaler_ANN = load('F:/Works/Application of ML/Final/scaler_ANN.joblib')

# Load the original data used during training
LSTM_cleaned_data = pd.read_csv('F:/Works/Application of ML/Final/LSTM_cleaned_data.csv')  # Load your original training data here
LSTM_cleaned_data

# Fit the scaler with the original data
scaler_LSTM.fit(LSTM_cleaned_data)

# Define the range of dates for which you want to make predictions
start_date = pd.to_datetime('2024-06-01')
end_date = pd.to_datetime('2024-06-30')

# Define the window size and features
window_size = 10  # Adjust this based on your model architecture
num_features = 6  # Adjust this based on the number of features in your input data

# Initialize the input data with dummy values (you should replace this with your actual data)
input_data = np.random.rand(window_size, num_features)

# Create a sequence of future dates within the specified range
future_dates = pd.date_range(start=start_date, end=end_date, freq='D')

# Create an empty array to store the predicted values
predicted_values = []

# Predict the next value for each day in the specified range
for _ in range(len(future_dates)):
    # Reshape the input data to match the model input shape
    input_data_reshaped = np.reshape(input_data, (1, input_data.shape[0], input_data.shape[1]))
    
    # Predict the next value using the loaded LSTM model
    next_value = loaded_lstm_model.predict(input_data_reshaped)[0]
    
    # Append the predicted value to the list of predicted values
    predicted_values.append(next_value)
    
    # Update the input data by removing the first time step and appending the predicted value
    input_data = np.append(input_data[1:], [next_value], axis=0)

# Inverse transform the predicted values to get the original scale
predicted_values = scaler_LSTM.inverse_transform(predicted_values)

# Create a DataFrame to store the predicted values along with corresponding dates
predicted_df = pd.DataFrame(predicted_values, columns=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'])
predicted_df['Date'] = future_dates

# Display the predicted DataFrame
predicted_df

# PM2.5 Sub-Index calculation
def get_PM25_subindex(x):
    if x <= 30:
        return x * 50 / 30
    elif x <= 60:
        return 50 + (x - 30) * 50 / 30
    elif x <= 90:
        return 100 + (x - 60) * 100 / 30
    elif x <= 120:
        return 200 + (x - 90) * 100 / 30
    elif x <= 250:
        return 300 + (x - 120) * 100 / 130
    elif x > 250:
        return 400 + (x - 250) * 100 / 130
    else:
        return 0

predicted_df["PM2.5_SubIndex"] = predicted_df["PM2.5"].apply(lambda x: get_PM25_subindex(x))

# PM10 Sub-Index calculation
def get_PM10_subindex(x):
    if x <= 50:
        return x
    elif x <= 100:
        return x
    elif x <= 250:
        return 100 + (x - 100) * 100 / 150
    elif x <= 350:
        return 200 + (x - 250)
    elif x <= 430:
        return 300 + (x - 350) * 100 / 80
    elif x > 430:
        return 400 + (x - 430) * 100 / 80
    else:
        return 0

predicted_df["PM10_SubIndex"] = predicted_df["PM10"].apply(lambda x: get_PM10_subindex(x))

# SO2 Sub-Index calculation
def get_SO2_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 380:
        return 100 + (x - 80) * 100 / 300
    elif x <= 800:
        return 200 + (x - 380) * 100 / 420
    elif x <= 1600:
        return 300 + (x - 800) * 100 / 800
    elif x > 1600:
        return 400 + (x - 1600) * 100 / 800
    else:
        return 0

predicted_df["SO2_SubIndex"] = predicted_df["SO2"].apply(lambda x: get_SO2_subindex(x))

# NO2 Sub-Index calculation
def get_NO2_subindex(x):
    if x <= 40:
        return x * 50 / 40
    elif x <= 80:
        return 50 + (x - 40) * 50 / 40
    elif x <= 180:
        return 100 + (x - 80) * 100 / 100
    elif x <= 280:
        return 200 + (x - 180) * 100 / 100
    elif x <= 400:
        return 300 + (x - 280) * 100 / 120
    elif x > 400:
        return 400 + (x - 400) * 100 / 120
    else:
        return 0

predicted_df["NO2_SubIndex"] = predicted_df["NO2"].apply(lambda x: get_NO2_subindex(x))

# CO Sub-Index calculation
def get_CO_subindex(x):
    if x <= 1:
        return x * 50 / 1
    elif x <= 2:
        return 50 + (x - 1) * 50 / 1
    elif x <= 10:
        return 100 + (x - 2) * 100 / 8
    elif x <= 17:
        return 200 + (x - 10) * 100 / 7
    elif x <= 34:
        return 300 + (x - 17) * 100 / 17
    elif x > 34:
        return 400 + (x - 34) * 100 / 17
    else:
        return 0

predicted_df["CO_SubIndex"] = predicted_df["CO"].apply(lambda x: get_CO_subindex(x))

# O3 Sub-Index calculation
def get_O3_subindex(x):
    if x <= 50:
        return x * 50 / 50
    elif x <= 100:
        return 50 + (x - 50) * 50 / 50
    elif x <= 168:
        return 100 + (x - 100) * 100 / 68
    elif x <= 208:
        return 200 + (x - 168) * 100 / 40
    elif x <= 748:
        return 300 + (x - 208) * 100 / 539
    elif x > 748:
        return 400 + (x - 400) * 100 / 539
    else:
        return 0

predicted_df["O3_SubIndex"] = predicted_df["O3"].apply(lambda x: get_O3_subindex(x))

# Assuming 'df' is your DataFrame
real_data = predicted_df.iloc[:, -6:]
real_data.head()

# Transform the new data using the loaded scaler
real_data_scaled = scaler_ANN.transform(real_data)
real_data_scaled

# Make predictions using the loaded model
predictions = loaded_ann_model.predict(real_data_scaled)
predictions

# Get the index of the class with the highest probability for each sample
predicted_labels = np.argmax(predictions, axis=1)
predicted_labels

# Define a dictionary mapping encoded numbers to original class labels
class_mappings = {
    0: 'Good',
    1: 'Satisfactory',
    2: 'Moderate',
    3: 'Poor',
    4: 'Very Poor',
    5: 'Severe'
}

# Replace the encoded numbers with their original class labels
predicted_class_labels = [class_mappings[label] for label in predicted_labels]

# Create a DataFrame to display the predicted class labels
predicted_df_ann = pd.DataFrame({'AQI Prediction': predicted_class_labels})
predicted_df_ann

first_six_columns_df = predicted_df.iloc[:, :7]
result_df = pd.concat([first_six_columns_df, predicted_df_ann], axis=1)
result_df

# Reorder columns with "Date" at the beginning
result_df = result_df[['Date', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'AQI Prediction']]


# Round numeric values to 2 decimal places
numeric_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
result_df[numeric_columns] = result_df[numeric_columns].round(2)
result_df

# Export DataFrame to a CSV file
result_df.to_csv('F:/Works/Application of ML/Final/prediction_ANN.csv', index=False)