import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report

# Load dataset
df = pd.read_csv("data/placementdata.csv")

# Drop unwanted columns
df.drop(["StudentID", "HSC_Marks"], axis=1, inplace=True)

# Encoding categorical columns
le = LabelEncoder()

df["ExtracurricularActivities"] = le.fit_transform(df["ExtracurricularActivities"])

df["PlacementTraining"] = le.fit_transform(df["PlacementTraining"])

df["PlacementStatus"] = le.fit_transform(df["PlacementStatus"])

# Features (inputs)
X = df.drop("PlacementStatus", axis=1)

# Target (output)
y = df["PlacementStatus"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

cm=confusion_matrix(y_test,y_pred)
print("confusion_matrix:")
print(cm)

print("\n classification report:")
print(classification_report(y_test,y_pred))


print("\n Feature importance:")
importance = model.feature_importances_

# Feature names
features = X.columns

# Display importance
for feature, score in zip(features, importance):
    print(feature, ":", score)

import joblib

# Save trained model
joblib.dump(model, "model/placement_model.pkl")

print("Model saved successfully!")