Create a complete, clean, professional README.md for my machine learning project called "Student Performance Predictor".

Use ONLY the information below. Do not invent any features, datasets, results, or claims.

Project:

* Python machine learning regression project
* Predicts final student score
* Dataset file: student_performance.csv
* Model file: student_performance_model.joblib
* Streamlit app: app.py

Input features:

* study_hours
* sleep_hours
* attendance
* previous_score
* assignment_score

Target:

* final_score

Models:

* Linear Regression
* Random Forest Regression

Linear Regression test results:

* MAE: 0.55
* RMSE: 0.69
* R²: 0.9981

Linear Regression 5-fold cross-validation:

* Mean MAE: 0.48
* Mean R²: 0.9945

Random Forest test results:

* MAE: 1.45
* RMSE: 2.10
* R²: 0.9824

Random Forest 5-fold cross-validation:

* Mean MAE: 1.05

Example student:

* Study hours: 5.5
* Sleep hours: 7
* Attendance: 88%
* Previous score: 75
* Assignment score: 82

Predictions:

* Linear Regression: 81.59 / 100
* Random Forest: 81.43 / 100

Dataset:

* 28 manually constructed student records
* Small and very clean dataset
* Educational/practice purpose only

Tech stack:

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* Streamlit
* Git
* GitHub

Include these sections:

1. Project title and short description
2. Features
3. Dataset
4. Models and Results
5. Example Prediction
6. How to Run
7. Project Structure
8. Tech Stack
9. Learning Goals
10. Disclaimer

Installation commands:

git clone https://github.com/badol2407-tech/student-performance-predictor.git
cd student-performance-predictor

python3 -m venv .venv
source .venv/bin/activate

pip install pandas numpy scikit-learn joblib streamlit

streamlit run app.py

Important:

* Return ONLY the complete README.md content.
* Use proper Markdown formatting.
* Keep it professional but beginner-friendly.
* Clearly mention that the dataset is manually constructed and too small to represent real-world predictive performance.
* Do not claim the model is production-ready.

