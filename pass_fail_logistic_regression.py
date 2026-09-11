import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = np.genfromtxt("students.csv", delimiter=",", skip_header=1)

all_features = data[:, :3]
final_scores = data[:, 3]

# Clean the missing study-hours value.
missing_rows, missing_columns = np.where(np.isnan(all_features))
column_means = np.nanmean(all_features, axis=0)
all_features[missing_rows, missing_columns] = column_means[missing_columns]

# Select two input features.
X = all_features[:, [0, 2]]

# Create a classification target: 1 = pass (70 or higher), 0 = not pass.
y = (final_scores >= 70).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Preprocessing + classification model.
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42),
)

# Training and evaluation.
model.fit(X_train, y_train)
test_predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, test_predictions)

# Inference for a new hypothetical student.
new_student = np.array([[6.0, 78.0]])
prediction = model.predict(new_student)[0]
probability = model.predict_proba(new_student)[0, 1]

print(f"Test accuracy: {accuracy:.2f}")
print(f"Predicted class (1=pass, 0=not pass): {prediction}")
print(f"Estimated pass probability: {probability:.2f}")
