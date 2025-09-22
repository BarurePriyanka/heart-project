# create_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
import joblib

# Generate dummy dataset
X, y = make_classification(n_samples=1000, n_features=5, random_state=42)

# Train a simple logistic regression model
model = LogisticRegression()
model.fit(X, y)

# Save the model
joblib.dump(model, "model.pkl")
print("model.pkl created successfully!")
