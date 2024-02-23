import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Read data
data = pd.read_csv('Booking_Revenue_Hotel_X.csv')

# Convert dates to datetime objects
data['StayDate'] = pd.to_datetime(data['StayDate'])
data['BookingDate'] = pd.to_datetime(data['BookingDate'])

# Aggregate data
aggregated_data_bookings = data.groupby(['RoomType', 'StayDate']).size().reset_index(name='NumBookings')
revenue_data = data.groupby(['StayDate']).agg({'Price': 'sum'}).reset_index()
revenue_data.rename(columns={'Price': 'Revenue'}, inplace=True)

# Merge bookings and revenue data
aggregated_data = pd.merge(aggregated_data_bookings, revenue_data, on='StayDate', how='left')

# Preprocess datetime data
aggregated_data['StayDayOfWeek'] = aggregated_data['StayDate'].dt.dayofweek
aggregated_data['StayMonth'] = aggregated_data['StayDate'].dt.month
aggregated_data['StayYear'] = aggregated_data['StayDate'].dt.year

# Prepare data for LSTM
room_types = pd.get_dummies(aggregated_data['RoomType'], drop_first=True)
X = pd.concat([aggregated_data[['StayDayOfWeek', 'StayMonth', 'StayYear']], room_types], axis=1)

y_bookings = aggregated_data['NumBookings']
y_revenue = aggregated_data['Revenue']

# Split data into train and test sets
#X_train, X_test, y_train_bookings, y_test_bookings, y_train_revenue, y_test_revenue = train_test_split(X, y_bookings, y_revenue, test_size=0.2, random_state=42)
X_train, X_test, y_train_bookings, y_test_bookings = train_test_split(X, y_bookings, test_size=0.2, random_state=42)
X_train_revenue, X_test_revenue, y_train_revenue, y_test_revenue = train_test_split(X, y_revenue, test_size=0.2, random_state=42)

# Scale the data
scaler_bookings = MinMaxScaler()
scaler_revenue = MinMaxScaler()

X_train_scaled_bookings = scaler_bookings.fit_transform(X_train)
X_test_scaled_bookings = scaler_bookings.transform(X_test)

X_train_scaled_revenue = scaler_revenue.fit_transform(X_train_revenue)
X_test_scaled_revenue = scaler_revenue.transform(X_test_revenue)

# Reshape data for LSTM (samples, timesteps, features)
X_train_reshaped_bookings = np.reshape(X_train_scaled_bookings, (X_train_scaled_bookings.shape[0], 1, X_train_scaled_bookings.shape[1]))
X_test_reshaped_bookings = np.reshape(X_test_scaled_bookings, (X_test_scaled_bookings.shape[0], 1, X_test_scaled_bookings.shape[1]))

X_train_reshaped_revenue = np.reshape(X_train_scaled_revenue, (X_train_scaled_revenue.shape[0], 1, X_train_scaled_revenue.shape[1]))
X_test_reshaped_revenue = np.reshape(X_test_scaled_revenue, (X_test_scaled_revenue.shape[0], 1, X_test_scaled_revenue.shape[1]))


# Define LSTM model for bookings
model_bookings = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train_reshaped_bookings.shape[1], X_train_reshaped_bookings.shape[2])),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)
])

# Compile the model
model_bookings.compile(optimizer='adam', loss='mean_squared_error')

# Train the model for bookings
model_bookings.fit(X_train_reshaped_bookings, y_train_bookings, epochs=10, batch_size=32, verbose=1)

# Make bookings predictions
predictions_bookings = model_bookings.predict(X_test_reshaped_bookings)
predictions_bookings = predictions_bookings.ravel()

# Evaluate bookings model
mse_bookings = tf.keras.metrics.mean_squared_error(y_test_bookings, predictions_bookings)
print("Mean Squared Error for Bookings:", mse_bookings)

# Define LSTM model for revenue
model_revenue = Sequential([
    LSTM(units=900, return_sequences=True, input_shape=(X_train_reshaped_revenue.shape[1], X_train_reshaped_revenue.shape[2])),
    Dropout(0.4),
    LSTM(units=600),
    Dropout(0.4),
    Dense(units=400, activation='relu'),
    Dropout(0.4),
    Dense(units=300, activation='relu'),
    Dropout(0.4),
    Dense(units=50, activation='relu'),
    Dense(units=1)
])

# Compile the model
model_revenue.compile(optimizer='adam', loss='mean_squared_error')

# Train the model for revenue
model_revenue.fit(X_train_reshaped_revenue, y_train_revenue, epochs=300, batch_size=32, verbose=1)
print(y_train_revenue[:10])

# Make revenue predictions
predictions_revenue = model_revenue.predict(X_test_reshaped_revenue)
print(predictions_revenue[:10])

#predictions_revenue = scaler_revenue.inverse_transform(predictions_revenue.reshape(-1, 1))
predictions_revenue = predictions_revenue.ravel()

# Evaluate revenue model
mse_revenue = tf.keras.metrics.mean_squared_error(y_test_revenue, predictions_revenue)
print("Mean Squared Error for Revenue:", mse_revenue)

# Filter data for the last 3 months (Nov 2023 to Jan 2024)
last_3_months_data = aggregated_data[(aggregated_data['StayDate'] >= '2023-11-01') & (aggregated_data['StayDate'] <= '2024-01-31')]

# Prepare data for LSTM
room_types_last_3_months = pd.get_dummies(last_3_months_data['RoomType'], drop_first=True)
X_last_3_months = pd.concat([last_3_months_data[['StayDayOfWeek', 'StayMonth', 'StayYear']], room_types_last_3_months], axis=1)

# Scale the data
X_last_3_months_scaled_bookings = scaler_bookings.transform(X_last_3_months)
X_last_3_months_scaled_revenue = scaler_revenue.transform(X_last_3_months)

# Reshape data for LSTM (samples, timesteps, features)
X_last_3_months_reshaped_bookings = np.reshape(X_last_3_months_scaled_bookings, (X_last_3_months_scaled_bookings.shape[0], 1, X_last_3_months_scaled_bookings.shape[1]))
X_last_3_months_reshaped_revenue = np.reshape(X_last_3_months_scaled_revenue, (X_last_3_months_scaled_revenue.shape[0], 1, X_last_3_months_scaled_revenue.shape[1]))

# Make predictions for bookings
predictions_last_3_months_bookings = model_bookings.predict(X_last_3_months_reshaped_bookings)
predictions_last_3_months_bookings = predictions_last_3_months_bookings.ravel()

# Make predictions for revenue
predictions_last_3_months_revenue = model_revenue.predict(X_last_3_months_reshaped_revenue)
predictions_last_3_months_revenue = predictions_last_3_months_revenue.ravel()

# Evaluate performance for bookings
mse_last_3_months_bookings = tf.keras.metrics.mean_squared_error(last_3_months_data['NumBookings'], predictions_last_3_months_bookings)
print("Mean Squared Error for Bookings (Last 3 Months):", mse_last_3_months_bookings)

# Evaluate performance for revenue
mse_last_3_months_revenue = tf.keras.metrics.mean_squared_error(last_3_months_data['Revenue'], predictions_last_3_months_revenue)
print("Mean Squared Error for Revenue (Last 3 Months):", mse_last_3_months_revenue)