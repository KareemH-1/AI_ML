
# Summary statistics for each course
courses = list(course_stats.keys())
summary_df = pd.DataFrame({
    'Course': courses,
    'Samples': [course_stats[c]['n_samples'] for c in courses],
    'R² Score': [f"{course_stats[c]['r2_score']:.4f}" for c in courses],
    'Study Hours Coef': [f"{course_stats[c]['coefficients']['study_hours']:.3f}" for c in courses],
    'Attendance Coef': [f"{course_stats[c]['coefficients']['class_attendance']:.3f}" for c in courses],
    'Sleep Hours Coef': [f"{course_stats[c]['coefficients']['sleep_hours']:.3f}" for c in courses]
})

print("\nModel Summary Statistics:")
print("="*80)
print(summary_df.to_string(index=False))
print("="*80)

# Residual plots to check model fit
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, course in enumerate(courses):
    course_data = data[data['course'] == course]
    X_course = course_data[numeric_features]
    y_actual = course_data['exam_score']
    y_pred = models[course].predict(X_course)
    y_pred = np.clip(y_pred, 0, 100)
    residuals = y_actual - y_pred
    
    axes[idx].scatter(y_pred, residuals, alpha=0.6, edgecolor='black')
    axes[idx].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[idx].set_title(f'{course.upper()} - Residual Plot', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Predicted Exam Score', fontsize=10)
    axes[idx].set_ylabel('Residuals', fontsize=10)
    axes[idx].grid(alpha=0.3)

# Hide the extra subplot
axes[5].axis('off')

plt.tight_layout()
plt.show()

# Visualize predictions for new students
plt.figure(figsize=(12, 6))
bars = plt.bar(range(len(new_students)), new_students['predicted_exam_score'], 
               color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
               edgecolor='black', linewidth=1.5)

# Add course labels and values on bars
for i, (course, score) in enumerate(zip(new_students['course'], new_students['predicted_exam_score'])):
    plt.text(i, score + 2, f'{score:.1f}', ha='center', fontsize=11, fontweight='bold')
    
plt.xticks(range(len(new_students)), new_students['course'].str.upper(), fontsize=11)
plt.title('Predicted Exam Scores for New Students', fontsize=14, fontweight='bold')
plt.xlabel('Course', fontsize=12)
plt.ylabel('Predicted Exam Score', fontsize=12)
plt.ylim(0, 110)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Display detailed predictions table
print("\nDetailed Predictions:")
print(new_students.to_string(index=False))

# Actual vs Predicted plots for each course
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, course in enumerate(courses):
    course_data = data[data['course'] == course]
    X_course = course_data[numeric_features]
    y_actual = course_data['exam_score']
    y_pred = models[course].predict(X_course)
    y_pred = np.clip(y_pred, 0, 100)
    
    axes[idx].scatter(y_actual, y_pred, alpha=0.6, edgecolor='black')
    axes[idx].plot([0, 100], [0, 100], 'r--', linewidth=2, label='Perfect Prediction')
    axes[idx].set_title(f'{course.upper()} - Actual vs Predicted', fontsize=11, fontweight='bold')
    axes[idx].set_xlabel('Actual Exam Score', fontsize=10)
    axes[idx].set_ylabel('Predicted Exam Score', fontsize=10)
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)
    axes[idx].set_xlim(0, 100)
    axes[idx].set_ylim(0, 100)

# Hide the extra subplot
axes[5].axis('off')

plt.tight_layout()
plt.show()

# Compare feature coefficients across courses
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, feature in enumerate(numeric_features):
    coefs = [course_stats[course]['coefficients'][feature] for course in courses]
    axes[idx].bar(courses, coefs, color='coral', edgecolor='black')
    axes[idx].set_title(f'Impact of {feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Course', fontsize=10)
    axes[idx].set_ylabel('Coefficient', fontsize=10)
    axes[idx].tick_params(axis='x', rotation=45)
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Visualize R² scores for each course model
courses = list(course_stats.keys())
r2_scores = [course_stats[course]['r2_score'] for course in courses]

plt.figure(figsize=(10, 6))
plt.bar(courses, r2_scores, color='steelblue', edgecolor='black')
plt.title('Model Performance (R² Score) by Course', fontsize=14, fontweight='bold')
plt.xlabel('Course', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.ylim(0, 1)
for i, (course, score) in enumerate(zip(courses, r2_scores)):
    plt.text(i, score + 0.02, f'{score:.3f}', ha='center', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
