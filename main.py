import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator

# ========================================================
# task1
print('task1')
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
print('task2')
# Identify the set of all names in the data, and sort these names in alphabetical order. How many are there? List the
# first and last 5 names.
names = stock_data['Name'].unique()
names.sort()
print('There are {} names in the data'.format(len(names)))
print('First 5 names: {}'.format(names[:5]))
print('Last 5 names: {}'.format(names[-5:]))
# ========================================================


# ========================================================
# task3
print('task3')
df = stock_data.copy()
df['first_date'] = pd.to_datetime(df['Date'])
df['last_date'] = pd.to_datetime(df['Date'])
df_original = df.copy()
df = df.groupby('Name').aggregate({'first_date': 'min', 'last_date': 'max'})
remove_data = df[(df['first_date'] > '2014-07-01') | (df['last_date'] < '2017-06-30')]
df = df.drop(remove_data.index)
print('Removed names: {}'.format(remove_data.index.values))
print('There are {} names left'.format(len(df)))
df = df_original[df_original['Name'].isin(df.index.values)]
# ========================================================


# ========================================================
# task4
print('task4')
remove_date = df[(df['Date'] < '2014-07-01') | (df['Date'] > '2017-06-30')]
df = df.drop(remove_date.index)
date_counts = df.groupby('Date')['Name'].nunique()
max_stock_count = df['Name'].nunique()
common_dates = date_counts[date_counts == max_stock_count].index
count_common_dates = len(common_dates)
first_5_dates = common_dates[:5]
last_5_dates = common_dates[-5:]
print("Number of common dates:", count_common_dates)
print("First 5 common dates:", first_5_dates.values)
print("Last 5 common dates:", last_5_dates.values)
df = df[df['Date'].isin(common_dates)]
# ========================================================


# ========================================================
# task5
print('task5')
daily_close = df.pivot(index='Date', columns='Name', values='Close')
print(daily_close)
# ========================================================


# ========================================================
# task6
print('task6')
daily_close_shift = daily_close.shift(1)
daily_close = (daily_close - daily_close_shift) / daily_close_shift
df_percent_change = daily_close.dropna()
print(df_percent_change)
# ========================================================


# ========================================================
# task7
print('task7')
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
print('task8')
# Get the explained variance ratios from the PCA object
explained_variance_ratios = pca.explained_variance_ratio_
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), explained_variance_ratios[:20], marker='o', linestyle='-', color='b')
plt.title('Explained Variance Ratios for Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)

# Find the elbow point
kneedle = KneeLocator(range(1, 21), explained_variance_ratios[:20], curve='convex', direction='decreasing')
plt.axvline(x=kneedle.knee, color='r', linestyle='--', label='Elbow')
plt.legend()
plt.show()
# ========================================================


# ========================================================
# task9
print('task9')
cumulative_variance_ratios = np.cumsum(explained_variance_ratios)
num_components_95 = np.where(cumulative_variance_ratios >= 0.95)[0][0] + 1
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(cumulative_variance_ratios) + 1), cumulative_variance_ratios, marker='o', linestyle='-',
         color='b')
plt.title('Cumulative Variance Ratios by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Variance Ratio')
plt.grid(True)
plt.axvline(x=num_components_95, color='r', linestyle='--', label=f'95% Variance at PC {num_components_95}')
plt.legend()
plt.show()
# ========================================================


# ========================================================
# task10
print('task10')
# Normalise your dataframe from step (6) so that the
# columns have zero mean and unit variance. Repeat steps (7) - (9) for this new dataframe.
scaler = StandardScaler()
normalized_data = scaler.fit_transform(df_percent_change)
normalized_df = pd.DataFrame(normalized_data, columns=df_percent_change.columns)
pca_normalized = PCA()
pca_normalized.fit(normalized_df)
eigenvalues_normalized = pca_normalized.explained_variance_
print("Top 5 Principal Components (Ranked by Eigenvalue):")
for i in range(5):
    print(f"PC {i+1}: Eigenvalue = {eigenvalues_normalized[i]}")
# Get the explained variance ratios from the PCA object
explained_variance_ratios_normalized = pca_normalized.explained_variance_ratio_
plt.figure(figsize=(10, 6))
plt.plot(range(1, 21), explained_variance_ratios_normalized[:20], marker='o', linestyle='-', color='b')
plt.title('Explained Variance Ratios for Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
# Find the elbow point
kneedle_normalized = KneeLocator(range(1, 21), explained_variance_ratios_normalized[:20], curve='convex', direction='decreasing')
plt.axvline(x=kneedle_normalized.knee, color='r', linestyle='--', label='Elbow')
plt.legend()
plt.show()
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
plt.axvline(x=num_components_95_normalized, color='r', linestyle='--', label=f'95% Variance at PC {num_components_95_normalized}')
plt.legend()
plt.grid(True)
plt.show()
