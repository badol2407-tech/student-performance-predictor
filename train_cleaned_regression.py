import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load raw structured data.
data = np.genfromtxt("students.csv", delimiter=",", skip_header=1)

# Separate input columns from the prediction target.
all_features = data[:, :3]
final_scores = data[:, 3]

# Clean missing values using each column's mean.
missing_rows, missing_columns = np.where(np.isnan(all_features))
column_means = np.nanmean(all_features, axis=0)
all_features[missing_rows, missing_columns] = column_means[missing_columns]

# Feature selection: hours studied and previous score.
X = all_features[:, [0, 2]]
y = final_scores

# Hold back 25% of data for evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Training.
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation.
test_predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)

# Inference for a new hypothetical student.
new_student = np.array([[6.0, 78.0]])
predicted_score = model.predict(new_student)[0]

print(f"Training rows: {len(X_train)}")
print(f"Evaluation rows: {len(X_test)}")
print(f"Test mean absolute error: {mae:.2f}")
print(f"Prediction for 6 study hours and previous score 78: {predicted_score:.1f}")
