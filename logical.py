import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
dataset = pd.read_csv('diabetes-2-1.csv')

# Split features (X) and target variable (y)
X = dataset.drop('Outcome', axis=1)
y = dataset['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)

# New observation values (assuming the same format as the dataset)
new_observation_values = [[7, 187, 50, 33, 392, 33.9, 0.826, 34]]

# Scale the new observation using the same scaler
new_observation_scaled = scaler.transform(new_observation_values)

# Predict the outcome for the new observation
predicted_outcome = model.predict(new_observation_scaled)
if predicted_outcome[0] == 0:
    print("Predicted outcome: Non-diabetic")
else:
    print("Predicted outcome: Diabetic")
