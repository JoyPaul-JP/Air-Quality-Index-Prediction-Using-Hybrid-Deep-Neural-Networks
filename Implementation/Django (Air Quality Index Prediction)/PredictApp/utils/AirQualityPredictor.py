import pandas as pd
import numpy as np
from keras.optimizers import Adam
from keras.models import load_model
from joblib import load, dump
from keras import backend as K
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

class AirQualityPredictor:
    def __init__(self, ann_model_path, lstm_model_path, scaler_ann_path, scaler_lstm_path, data_path):
        self.ann_model_path = ann_model_path
        self.lstm_model_path = lstm_model_path
        self.scaler_ann_path = scaler_ann_path
        self.scaler_lstm_path = scaler_lstm_path
        self.data_path = data_path
        
        # Load the models
        self.loaded_ann_model = load_model(self.ann_model_path)
        self.loaded_lstm_model = load_model(self.lstm_model_path, custom_objects={'mse': self.mse})
        
        # Compile the ANN model
        self.loaded_ann_model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Load the scalers
        self.scaler_ann = load(self.scaler_ann_path)
        self.scaler_lstm = load(self.scaler_lstm_path)
        
        # Load the original data used during training
        self.original_data = pd.read_csv(self.data_path)
        
        # Fit the LSTM scaler with the original data
        self.scaler_lstm.fit(self.original_data)

    def mse(self, y_true, y_pred):
        return K.mean(K.square(y_pred - y_true), axis=-1)
    
    def calculate_subindices(self, df):
        df["PM2.5_SubIndex"] = df["PM2.5"].apply(self.get_pm25_subindex)
        df["PM10_SubIndex"] = df["PM10"].apply(self.get_pm10_subindex)
        df["SO2_SubIndex"] = df["SO2"].apply(self.get_so2_subindex)
        df["NO2_SubIndex"] = df["NO2"].apply(self.get_no2_subindex)
        df["CO_SubIndex"] = df["CO"].apply(self.get_co_subindex)
        df["O3_SubIndex"] = df["O3"].apply(self.get_o3_subindex)
        return df
    
    def predict_lstm(self, input_data, future_dates):
        # Predict the next values using LSTM model
        predicted_values = []
        for _ in range(len(future_dates)):
            # Reshape the input data to match the model input shape
            input_data_reshaped = np.reshape(input_data, (1, input_data.shape[0], input_data.shape[1]))
            
            # Predict the next value using the loaded LSTM model
            next_value = self.loaded_lstm_model.predict(input_data_reshaped)[0]
            
            # Append the predicted value to the list of predicted values
            predicted_values.append(next_value)
            
            # Update the input data by removing the first time step and appending the predicted value
            input_data = np.append(input_data[1:], [next_value], axis=0)
        
        # Inverse transform the predicted values to get the original scale
        predicted_values = self.scaler_lstm.inverse_transform(predicted_values)
        
        # Create a DataFrame to store the predicted values along with corresponding dates
        predicted_df = pd.DataFrame(predicted_values, columns=['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'])
        predicted_df['Date'] = future_dates
        
        return predicted_df
    
    def get_pm25_subindex(self, x):
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

    def get_pm10_subindex(self, x):
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

    def get_so2_subindex(self, x):
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

    def get_no2_subindex(self, x):
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

    def get_co_subindex(self, x):
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

    def get_o3_subindex(self, x):
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
    
    def predict_ann(self, real_data):
        # Transform the new data using the loaded scaler
        real_data_scaled = self.scaler_ann.transform(real_data)
        
        # Make predictions using the loaded model
        predictions = self.loaded_ann_model.predict(real_data_scaled)
        
        # Get the index of the class with the highest probability for each sample
        predicted_labels = np.argmax(predictions, axis=1)
        
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
        
        return predicted_df_ann

    def run_predictions(self, start_date, end_date):
        # Define the range of dates for which you want to make predictions
        future_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Define the window size and features
        window_size = 10  # Adjust this based on your model architecture
        num_features = 6  # Adjust this based on the number of features in your input data
        
        # Initialize the input data with dummy values (you should replace this with your actual data)
        input_data = np.random.rand(window_size, num_features)
        
        # Predict future values using the LSTM model
        predicted_df = self.predict_lstm(input_data, future_dates)
        
        # Calculate sub-indices
        predicted_df = self.calculate_subindices(predicted_df)
        
        # Use the last 6 columns as input to the ANN model
        real_data = predicted_df.iloc[:, -6:]
        
        # Predict AQI class labels using the ANN model
        predicted_df_ann = self.predict_ann(real_data)
        
        # Combine the results
        first_six_columns_df = predicted_df.iloc[:, :7]
        result_df = pd.concat([first_six_columns_df, predicted_df_ann], axis=1)
        
        # Reorder columns with "Date" at the beginning
        result_df = result_df[['Date', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3', 'AQI Prediction']]
        
        # Round numeric values to 2 decimal places
        numeric_columns = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
        result_df[numeric_columns] = result_df[numeric_columns].round(2)
        
        return result_df
    
    def save_predictions(self, result_df, output_path):
        # Export DataFrame to a CSV file
        result_df.to_csv(output_path, index=False)