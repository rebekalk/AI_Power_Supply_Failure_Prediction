import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load the dataset
data = pd.read_csv("data/power_supply_data.csv")

# Select input parameters
X = data[
    ["input_voltage", "output_voltage", "current", "temperature"]
]

# Select target output
y = data["status"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Create the AI model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

# Display evaluation results
print("Model Accuracy:", accuracy_score(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

# Save the trained model
with open("power_supply_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\nModel trained and saved successfully!")
