import numpy as np

# Load numeric CSV data; the empty cell becomes NaN (missing value).
data = np.genfromtxt("students.csv", delimiter=",", skip_header=1)

features = data[:, :3]
target = data[:, 3]

# Cleaning: replace missing values with the mean of that column.
missing_rows, missing_columns = np.where(np.isnan(features))
column_means = np.nanmean(features, axis=0)
features[missing_rows, missing_columns] = column_means[missing_columns]

# Feature selection: use study hours and previous score to predict final score.
selected_features = features[:, [0, 2]]

print(f"Missing values after cleaning: {np.isnan(features).sum()}")
print(f"Selected feature matrix shape: {selected_features.shape}")
print(f"Target values: {target}")
