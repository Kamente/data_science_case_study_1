import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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

# Get feature names
feature_names = X.columns.tolist()

# Train the Support Vector Classifier model
model = SVC(kernel='linear', random_state=42)  
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:")
print(conf_matrix)

# New observation values
new_observation_values = [[2, 105, 75, 0, 0, 23.3, 0.56, 53]]

# Scale the new observation 
new_observation_scaled = scaler.transform(new_observation_values)

# Predict the outcome for the new observation
predicted_outcome = model.predict(new_observation_scaled)
if predicted_outcome[0] == 0:
    print("Predicted outcome: Non-diabetic")
else:
    print("Predicted outcome: Diabetic")
