import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

# ========================================================
# task1
print('Task1')
# Read the data from the csv file
stock_data = pd.read_csv('stock_data.csv')
# Name: The stock name
stock_data.rename(columns={'date': 'Date'}, inplace=True)
stock_data.rename(columns={'open': 'Open'}, inplace=True)
stock_data.rename(columns={'high': 'High'}, inplace=True)
stock_data.rename(columns={'low': 'Low'}, inplace=True)
stock_data.rename(columns={'close': 'Close'}, inplace=True)
stock_data.rename(columns={'volume': 'Volume'}, inplace=True)
stock_data.rename(columns={'Name': 'Name'}, inplace=True)
# ========================================================


# ========================================================
# task2
print('Task2')
# Get the unique names of the stocks
names = stock_data['Name'].unique()

# Sort the names alphabetically
names.sort()

# Print the first five names
print('There are {} names in the data'.format(len(names)))
print('First 5 names: {}'.format(names[:5]))
print('Last 5 names: {}'.format(names[-5:]))
# ========================================================


# ========================================================
# task3
print('Task3')

df = stock_data.copy()
df_original = df.copy()

# convert the date column to datetime
df['first_date'] = pd.to_datetime(df['Date'])
df['last_date'] = pd.to_datetime(df['Date'])

# get the first and last date for each stock
df = df.groupby('Name').aggregate({'first_date': 'min', 'last_date': 'max'})

# remove stocks that do not have data for the entire period
remove_data = df[(df['first_date'] > '2014-07-01') | (df['last_date'] < '2017-06-30')]
df = df.drop(remove_data.index)

# print the names of the removed stocks
print('Removed names: {}'.format(remove_data.index.values))

# print the number of names left
print('There are {} names left'.format(len(df)))

# get the names of the stocks that are left
df = df_original[df_original['Name'].isin(df.index.values)]
# ========================================================


# ========================================================
# task4
print('Task4')

# remove stocks that do not have data for the entire period
remove_date = df[(df['Date'] < '2014-07-01') | (df['Date'] > '2017-06-30')]
df = df.drop(remove_date.index)

# get the number of common dates
date_counts = df.groupby('Date')['Name'].nunique()
max_stock_count = df['Name'].nunique()

# get the stock names for the common dates
common_dates = date_counts[date_counts == max_stock_count].index

# get the first and last 5 common dates
count_common_dates = len(common_dates)
first_5_dates = common_dates[:5]
last_5_dates = common_dates[-5:]

# print the results
print("Number of common dates:", count_common_dates)
print("First 5 common dates:", first_5_dates.values)
print("Last 5 common dates:", last_5_dates.values)

# filter the data for the common dates
df = df[df['Date'].isin(common_dates)]
# ========================================================


# ========================================================
# task5
print('Task5')

# Pivot the data so that the stock names are the columns, the dates are the rows, and the values are the closing prices
daily_close = df.pivot(index='Date', columns='Name', values='Close')

# Print the first and last five rows of daily_close
print(daily_close)
# ========================================================


# ========================================================
# task6
print('Task6')

# Calculate the daily percentage change for `daily_close`
daily_close_shift = daily_close.shift(1)
daily_close = (daily_close - daily_close_shift) / daily_close_shift

# Remove the NaN values from daily_close
df_percent_change = daily_close.dropna()

# Print the first five rows of df_percent_change
print(df_percent_change)
# ========================================================


# ========================================================
# task7
print('Task7')

# Create a PCA model with 20 components
pca = PCA()
pca.fit(df_percent_change)

# Get the eigenvalues (explained variance) and sort them in descending order
eigenvalues = pca.explained_variance_
sorted_indices = eigenvalues.argsort()[::-1]

# Print the top five principal components according to their eigenvalues
print("Top 5 Principal Components (Ranked by Eigenvalue):")
for i in range(5):
    index = sorted_indices[i]
    print(f"PC {i + 1}: Eigenvalue = {eigenvalues[index]}")
# ========================================================


# ========================================================
# task8
print('Task8')

# Get the explained variance ratios from the PCA object
explained_variance_ratios = pca.explained_variance_ratio_

# Plot the explained variance ratios
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), explained_variance_ratios[:20], marker='o', linestyle='-', color='b')
plt.title('Explained Variance Ratios for Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)

# Find the elbow point
kneedle = KneeLocator(range(1, 21), explained_variance_ratios[:20], curve='convex', direction='decreasing')
plt.axvline(x=kneedle.knee, color='r', linestyle='--', label='Elbow')

# Show the plot
plt.legend()
plt.show()

# percentage of variance is explained by the first principal component
print('Percentage of variance is explained by the first principal component: {}%'.format(explained_variance_ratios[0]*100))
# ========================================================


# ========================================================
# task9
print('Task9')

# Calculate cumulative variance ratios
cumulative_variance_ratios = np.cumsum(explained_variance_ratios)

# Find the number of components needed to reach 95% cumulative variance
num_components_95 = np.where(cumulative_variance_ratios >= 0.95)[0][0] + 1

# Plot the cumulative variance ratios
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(cumulative_variance_ratios) + 1), cumulative_variance_ratios, marker='o', linestyle='-',
         color='b')
plt.title('Cumulative Variance Ratios by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Ratio')
plt.grid(True)

# Plot a vertical line at the number of components needed for 95% variance
plt.axvline(x=num_components_95, color='r', linestyle='--', label=f'95% Variance at PC {num_components_95}')

# Show the plot
plt.legend()
plt.show()
# ========================================================


# ========================================================
# task10
print('Task10')

# standardize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df_percent_change)
normalized_df = pd.DataFrame(normalized_data, columns=df_percent_change.columns)

# Create a PCA model and fit it with the standardized data
pca_normalized = PCA()
pca_normalized.fit(normalized_df)

# Get the eigenvalues (explained variance) and sort them in descending order
eigenvalues_normalized = pca_normalized.explained_variance_

# Print the top five principal components according to their eigenvalues
print("Top 5 Principal Components (Ranked by Eigenvalue):")
for i in range(5):
    print(f"PC {i+1}: Eigenvalue = {eigenvalues_normalized[i]}")

# Get the explained variance ratios from the PCA object
explained_variance_ratios_normalized = pca_normalized.explained_variance_ratio_

# Plot the explained variance ratios
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), explained_variance_ratios_normalized[:20], marker='o', linestyle='-', color='b')
plt.title('Explained Variance Ratios for Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)

# Find the elbow point
kneedle_normalized = KneeLocator(range(1, 21), explained_variance_ratios_normalized[:20], curve='convex', direction='decreasing')
plt.axvline(x=kneedle_normalized.knee, color='r', linestyle='--', label='Elbow')

# Show the plot
plt.legend()
plt.show()

# percentage of variance is explained by the first principal component
print('Percentage of variance is explained by the first principal component: {}%'.format(explained_variance_ratios_normalized[0]*100))

# Calculate cumulative variance ratios
cumulative_variance_ratios_normalized = np.cumsum(explained_variance_ratios_normalized)

# Find the number of components needed to reach 95% cumulative variance
num_components_95_normalized = np.where(cumulative_variance_ratios_normalized >= 0.95)[0][0] + 1

# Plot the cumulative variance ratios
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(cumulative_variance_ratios_normalized) + 1), cumulative_variance_ratios_normalized, marker='o', linestyle='-', color='b')
plt.title('Cumulative Variance Ratios by Principal Components (Normalized Data)')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Ratio')

# Plot a vertical line at the number of components needed for 95% variance
plt.axvline(x=num_components_95_normalized, color='r', linestyle='--', label=f'95% Variance at PC {num_components_95_normalized}')

# Show the plot
plt.legend()
plt.grid(True)
plt.show()
