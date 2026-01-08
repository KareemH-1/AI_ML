# Supervised Learning with scikit-Learn


# Import kagglehub (commented out)
#import kagglehub
#path = kagglehub.dataset_download("kundanbedmutha/exam-score-prediction-dataset", path=".")
#print("Path to dataset files:", path)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("./Datasets/Exam_Score_Prediction.csv")
print(data.head())

data = data.dropna()

print("gender: ", data['gender'].value_counts(), '\n')
print("course: ", data['course'].value_counts(), '\n')
print("internet_access", data["internet_access"].value_counts(), '\n')
print("sleep_quality: ", data['sleep_quality'].value_counts(), '\n')
print("study_method: ", data['study_method'].value_counts(), '\n')
print("facility_rating: ", data['facility_rating'].value_counts(), '\n')
print("exam_difficulty: ", data['exam_difficulty'].value_counts())

# Keep only numeric features for regression
# Categorical features (gender, course, study_method) will be handled by separate models per course
print("Unique courses:", data['course'].unique())
print("\nData shape:", data.shape)
print(data.head())

# Visualize the distribution of exam scores
plt.hist(data['exam_score'], bins=10, edgecolor='black')
plt.title('Distribution of Exam Scores')
plt.xlabel('Exam Score')
plt.ylabel('Number of Students')
plt.show()

# Select only numeric features for regression
numeric_features = ['study_hours', 'class_attendance', 'sleep_hours']

print(f"Using numeric features: {numeric_features}")
print(f"\nData types:\n{data[numeric_features + ['exam_score']].dtypes}")

# Plot the relationship between study hours and exam score
plt.scatter(data['study_hours'], data['exam_score'])
plt.title('Study Hours vs Exam Score')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.show()
# Output is linear

plt.scatter(data['class_attendance'], data['exam_score'])
plt.title('Class Attendance vs Exam Score')
plt.xlabel('Class Attendance (%)')
plt.ylabel('Exam Score')
plt.show()

plt.scatter(data['sleep_hours'], data['exam_score'])
plt.title('Sleep Hours vs Exam Score')
plt.xlabel('Sleep Hours')
plt.ylabel('Exam Score')
plt.show()

from sklearn.linear_model import LinearRegression

# Build separate models for each course
models = {}
course_stats = {}

for course in data['course'].unique():
    # Filter data for this course
    course_data = data[data['course'] == course]
    
    X_course = course_data[numeric_features]
    y_course = course_data['exam_score']
    
    # Train model
    model = LinearRegression()
    model.fit(X_course, y_course)
    
    # Store model and stats
    models[course] = model
    course_stats[course] = {
        'n_samples': len(course_data),
        'r2_score': model.score(X_course, y_course),
        'coefficients': dict(zip(numeric_features, model.coef_)),
        'intercept': model.intercept_
    }
    
    print(f"\n{'='*50}")
    print(f"Course: {course}")
    print(f"Samples: {course_stats[course]['n_samples']}")
    print(f"R² Score: {course_stats[course]['r2_score']:.4f}")
    print(f"Intercept: {course_stats[course]['intercept']:.2f}")
    print(f"Coefficients:")
    for feature, coef in course_stats[course]['coefficients'].items():
        print(f"  {feature}: {coef:.4f}")

# Predict on new data using course-specific models
new_students = pd.DataFrame({
    'course': ['b.tech', 'b.sc', 'ba', 'b.com', 'bca'],
    'study_hours': [12.0, 10.0, 5.0, 8.0, 11.0],
    'class_attendance': [70.0, 60.0, 70.0, 10.0, 95.0],
    'sleep_hours': [7.0, 6.0, 5.0, 6.5, 7.5]
})

predictions = []

for idx, row in new_students.iterrows():
    course = row['course']
    
    if course in models:
        X_new = pd.DataFrame([row[numeric_features]], columns=numeric_features)
        pred = models[course].predict(X_new)[0]
        pred = np.clip(pred, 0, 100)
        predictions.append(pred)
    else:
        predictions.append(np.nan)

new_students['predicted_exam_score'] = np.round(predictions, 2)

print("Predictions using course-specific models:")
print(new_students)

# plot predicted exam scores for new students
plt.bar(new_students['course'], new_students['predicted_exam_score'], color='skyblue')
plt.title('Predicted Exam Scores for New Students')
plt.xlabel('Course')
plt.ylabel('Predicted Exam Score')
plt.show()