import pandas as pd

# Load historical bookings and reservations data
data = pd.read_csv("D:\Tensorflow\PerformanceAnalysis\Booking_Revenue_Hotel_X.csv")

# Convert dates to datetime objects
data['StayDate'] = pd.to_datetime(data['StayDate'])
data['BookingDate'] = pd.to_datetime(data['BookingDate'])

# Separate data for previous and current years
previous_year_data = data[data['StayDate'].dt.year == 2023]
current_year_data = data[data['StayDate'].dt.year == 2024]

# Calculate monthly revenue for current year (2024)
monthly_revenue_2024 = current_year_data.groupby(current_year_data['StayDate'].dt.month)['Price'].sum()
print(monthly_revenue_2024)
# Calculate monthly revenue for previous year (2023)
monthly_revenue_2023 = previous_year_data.groupby(previous_year_data['StayDate'].dt.month)['Price'].sum()
print(monthly_revenue_2023)
# Compare monthly revenue between 2024 and 2023
revenue_comparison = monthly_revenue_2024 / monthly_revenue_2023 - 1

# Calculate Average Daily Rate (ADR) for each room type
previous_year_data['LengthOfStay'] = (previous_year_data['StayDate'] - previous_year_data['BookingDate']).dt.days
previous_year_data['ADR'] = previous_year_data['Price'] / previous_year_data['LengthOfStay']
adr_by_room_type = previous_year_data.groupby(['RoomType', previous_year_data['StayDate'].dt.month])['ADR'].mean()

# Calculate Occupancy Rate for each month
total_room_nights_sold = previous_year_data.groupby(previous_year_data['StayDate'].dt.month)['LengthOfStay'].sum()
#occupancy_rate = total_room_nights_sold / total_available_room_nights  # Assuming total_available_room_nights is known

# Calculate Booking Lead Time statistics
booking_lead_time = (previous_year_data['StayDate'] - previous_year_data['BookingDate']).dt.days
booking_lead_time_stats = booking_lead_time.describe()

# Room Type Performance Analysis
room_type_performance = previous_year_data.groupby(['RoomType', previous_year_data['StayDate'].dt.month]).agg({'Price': 'sum', 'LengthOfStay': 'sum'})
room_type_performance['ADR'] = room_type_performance['Price'] / room_type_performance['LengthOfStay']
#room_type_performance['OccupancyRate'] = room_type_performance['LengthOfStay'] / total_available_room_nights

# Revenue Forecasting (Example: Using simple average)
average_monthly_revenue_2024 = monthly_revenue_2024.mean()

# Print KPIs and analysis results
print("Monthly Revenue Comparison (2024 vs 2023):")
print(revenue_comparison)

print("\nAverage Daily Rate (ADR) by Room Type:")
print(adr_by_room_type)

print("\nOccupancy Rate by Month:")
#print(occupancy_rate)

print("\nBooking Lead Time Statistics:")
print(booking_lead_time_stats)

print("\nRoom Type Performance Analysis:")
print(room_type_performance)

print("\nRevenue Forecast for Remaining Months of 2024:")
print("Average Monthly Revenue (2024):", average_monthly_revenue_2024)