import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso
from datetime import datetime

# Read data
data = pd.read_csv('Booking_Revenue_Hotel_X.csv')

# Convert dates to datetime objects
data['StayDate'] = pd.to_datetime(data['StayDate'])
data['BookingDate'] = pd.to_datetime(data['BookingDate'])

# Aggregate data
aggregated_data = data.groupby(['RoomType', 'StayDate']).size().reset_index(name='NumBookings')

# Perform one-hot encoding for categorical variables
aggregated_data_encoded = pd.get_dummies(aggregated_data, columns=['RoomType'])
aggregated_data_encoded[['RoomType_Room 1', 'RoomType_Room 2', 'RoomType_Room 3', 'RoomType_Room 4']] = aggregated_data_encoded[['RoomType_Room 1', 'RoomType_Room 2', 'RoomType_Room 3', 'RoomType_Room 4']].astype(int)

# Preprocess datetime data
aggregated_data_encoded['StayDayOfWeek'] = aggregated_data_encoded['StayDate'].dt.dayofweek
aggregated_data_encoded['StayMonth'] = aggregated_data_encoded['StayDate'].dt.month
aggregated_data_encoded['StayYear'] = aggregated_data_encoded['StayDate'].dt.year

revenue_data = data.groupby(['StayDate']).agg({'Price': 'sum'}).reset_index()
print(revenue_data.shape)
print(aggregated_data_encoded.shape)
revenue_data.rename(columns={'Price': 'Revenue'}, inplace=True)
aggregated_data_encoded = pd.merge(aggregated_data_encoded, revenue_data, on='StayDate', how='left')
#revenue_data.rename(columns={'Price': 'Revenue'}, inplace=True)

aggregated_data_encoded.drop(columns=['StayDate'], inplace=True)

if not aggregated_data_encoded['Revenue'].isna().any():
    print("None of the elements in column 'B' is NaN")
else:
    print("There are NaN values in column 'B'")

# Implement forecasting algorithm
def forecast_bookings(data):
    # Split data into features and target
    X = data.drop(columns=['NumBookings'])
    y = data['NumBookings']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X.columns)
    print(X_train.shape)
    print(y_train.shape)

    # Initialize and train the model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Make predictions
    predictions_n = model.predict(X_test.iloc[0:2, :])
    print(y_test.iloc[0:2])
    print(predictions_n)
    predictions = model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error:", mse)
    return model

def forecast_revenue(data):
    # Split data into features and target
    #X = data.drop(columns=['NumBookings', 'Revenue'])

    X = data.drop(columns=['Revenue'])
    y = data['Revenue']

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = GradientBoostingRegressor()
    print(X_train.columns)
    model.fit(X_train, y_train)
    predictions_n = model.predict(X_train.iloc[8:11, :])
    print(y_train.iloc[8:11])
    print(predictions_n)

    # Make predictions
    #predictions_n = model.predict(X_test.iloc[0:2, :])
    #print(y_test.iloc[0:2])
    #print(predictions_n)
    predictions = model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, predictions)
    print("Mean Squared Error for Revenue:", mse)
    return model

# Perform forecasting
#model = forecast_bookings(aggregated_data_encoded)
model = forecast_revenue(aggregated_data_encoded)