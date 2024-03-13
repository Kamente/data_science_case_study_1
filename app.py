import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# import the dataset
dataset = pd.read_csv('diabetes-2-1.csv')

# Extract features of the disease (first 8 columns) and target variable (last column)
X = dataset.iloc[:, :8].values
y = dataset.iloc[:, -1].values

# Training and testing the data sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# Train the model
regressor = RandomForestRegressor(n_estimators=10, random_state=0)
regressor.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

print("Mean Squared Error:", mse)

# Creating a new observation with the first 8 features
new_observation_values = [1,168,88,29,0,35,0.905,52]
new_observation = [new_observation_values]


# Predict the outcomes using the new observation
print("Predicted Outcome in the next 5 years:")
predicted_outcome = regressor.predict(new_observation)
print("Predicted Outcome:", predicted_outcome)

# Load outcomes from the dataset
actual_outcomes = dataset['Outcome']

# Compare the predicted outcomes with the actual outcomes
for i, pred in enumerate(predicted_outcome):
    actual = actual_outcomes[i]
    if pred >= 0.5:
        print(f"Predicted Outcome : 1 (Diabetic)")
    else:
        print(f"Predicted Outcome: 0 (Non-diabetic)")
