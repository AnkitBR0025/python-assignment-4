import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Step 1: Load Data from CSV ---
print("Loading data from sample_weather_data.csv ...")
df = pd.read_csv('sample_weather_data.csv')

# Check first few rows and info
print(df.head())
print(df.info())

# --- Step 2: Clean Data ---
# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Fill missing values
# For Temperature, use average; for Rainfall, assume 0
mean_temp = df['Temperature'].mean()
df['Temperature'] = df['Temperature'].fillna(mean_temp)
df['Rainfall'] = df['Rainfall'].fillna(0)

print("\nData cleaned. Missing values handled.")

# --- Step 3: Statistics with NumPy ---
print("\n--- Statistics ---")
temps = df['Temperature'].to_numpy()
rain = df['Rainfall'].to_numpy()

print("Average Temp:", np.mean(temps))
print("Max Temp:", np.max(temps))
print("Min Temp:", np.min(temps))
print("Total Rainfall:", np.sum(rain))

# --- Step 4: Grouping by Month ---
df['Month'] = df['Date'].dt.month_name()

# Group by month to see average temp and total rain
grouped = df.groupby('Month')[['Temperature', 'Rainfall']].mean()
print("\nMonthly Averages:")
print(grouped)

# --- Step 5: Visualization ---

print("\nCreating plots...")

# Plot 1: Line chart for Temperature
plt.figure(figsize=(10, 5))
plt.plot(df['Date'], df['Temperature'], color='red')
plt.title('Daily Temperature')
plt.xlabel('Date')
plt.ylabel('Temp (C)')
plt.savefig('temp_chart.png')
plt.show()

# Plot 2: Scatter plot (Humidity vs Temp)
plt.figure()
plt.scatter(df['Temperature'], df['Humidity'])
plt.title('Temp vs Humidity')
plt.xlabel('Temp')
plt.ylabel('Humidity')
plt.savefig('scatter_plot.png')
plt.show()

# Plot 3: Combined plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
ax1.plot(df['Date'], df['Temperature'], 'r-')
ax1.set_title('Temperature Trend')
ax1.set_ylabel('Temp (C)')

ax2.bar(df['Date'], df['Rainfall'], color='blue')
ax2.set_title('Rainfall')
ax2.set_ylabel('Rain (mm)')
ax2.set_xlabel('Date')

plt.tight_layout()
plt.savefig('combined_plot.png')
plt.show()

# --- Step 6: Export cleaned data ---
df.to_csv('final_cleaned_data.csv', index=False)
print("Analysis done! Saved final_cleaned_data.csv and plot images.")
