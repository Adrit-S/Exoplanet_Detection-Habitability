import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the dataset
DATA_DIRECTORY = r"D:\Exoplanet Dataset\Confirmed_Exoplanet_Parameters.csv.csv"
df = pd.read_csv(DATA_DIRECTORY, comment='#', skiprows=9)

# Define Earth reference values
earth_params = {
    "radius": 1.0,       # Earth radii
    "teq": 288,          # Earth equilibrium temperature 
    "insolation": 1.0,   # Earth flux
    "steff": 5778        # Sun temperature 
}

# Features for habitability calculation
features = ["koi_prad", "koi_teq", "koi_insol", "koi_steff"]
weights = [0.57, 5.0, 2.0, 1.0]  # Corresponding weights based on ESI index
# Mapping from dataset column names to earth_params keys
feature_map = {
    "koi_prad": "radius",
    "koi_teq": "teq",
    "koi_insol": "insolation",
    "koi_steff": "steff"
}
# Trim rows missing necessary parameters
df = df.dropna(subset=features)

# Function to compute habitability score using ESI based formula
def calculate_habitability(row):
    esi_components = []
    for i, feature in enumerate(features):
        x_i = row[feature]
        x_earth = earth_params[feature_map[feature]]  # Use the mapped key
        w_i = weights[i]
        esi = (1 - abs((x_i - x_earth) / (x_i + x_earth))) ** w_i
        esi_components.append(esi)
    return np.prod(esi_components) * 100  # Scale to 0-100%

# Compute habitability scores
df["habitability_score"] = df.apply(calculate_habitability, axis=1)

# Split data into features (X) and target (y)
X = df[features]
y = df["habitability_score"]

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)

# Predict on test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")

# Save the trained model
model_path = r"D:\Exoplanet Dataset\habitability_model.pkl"
joblib.dump(rf_model, model_path)
print(f"Model saved to {model_path}")

# Visualize one tree from the random forest
plt.figure(figsize=(1, 0.5))  # Adjust the size as needed
plot_tree(rf_model.estimators_[0],  # Access the first decision tree
          filled=True,              # Fill nodes with colors based on majority class
          feature_names=features,  # Names of the features
          rounded=True,             # Round the corners of the nodes
          fontsize=1)             # Set the font size for readability
plt.show()

# Plot Predicted vs Actual values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')  # Line of perfect prediction
plt.title(f"Predicted vs Actual Habitability Scores (R² = {r2:.2f})")
plt.xlabel("Actual Habitability Score")
plt.ylabel("Predicted Habitability Score")
plt.show()
