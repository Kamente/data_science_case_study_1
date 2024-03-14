import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# import the dataset
dataset = pd.read_csv('diabetes-2-1.csv')

# Separate features (first 8 columns) and target variable (last column)
X = dataset.iloc[:, :8].values
y = dataset.iloc[:, -1].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Train the model
classifier = RandomForestClassifier(n_estimators=10, random_state=0)
classifier.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Creating a new observation with the first 8 features
new_observation_values = [2, 105, 75, 0, 0, 23.3, 0.56, 53]

# Predict the outcome using the new observation
predicted_outcome = classifier.predict([new_observation_values])

# **Confusion Matrix for overall model performance**
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Print predicted outcome for the new observation
if predicted_outcome == 1:
    print("Predicted Outcome: Diabetic")
else:
    print("Predicted Outcome: Non-diabetic")
