import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Training data: study hours (feature) and exam scores (target)
X_train = np.array([[1], [2], [3], [4], [5], [6]])
y_train = np.array([48, 55, 62, 68, 75, 82])

# Evaluation data kept separate from training
X_test = np.array([[2.5], [4.5], [6.5]])
y_test = np.array([58, 72, 86])

# Training: learn the relationship from training data
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation: compare predictions with known test scores
test_predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)

# Inference: predict a score for a new input
new_student_hours = np.array([[5.5]])
predicted_score = model.predict(new_student_hours)[0]

print(f"Mean absolute error: {mae:.2f}")
print(f"Predicted score for 5.5 study hours: {predicted_score:.1f}")
