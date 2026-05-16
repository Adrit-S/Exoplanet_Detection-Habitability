import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# Load the dataset
DATA_DIRECTORY = r"D:\Exoplanet Dataset\Confirmed_Exoplanet_Parameters.csv.csv"
df = pd.read_csv(DATA_DIRECTORY)

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

# Trim rows missing necessary parameters
df = df.dropna(subset=features)

# Function to compute habitability score using ESI based formula
def calculate_habitability(row):
    esi_components = []
    for i, feature in enumerate(features):
        x_i = row[feature]
        x_earth = earth_params[feature.split("_")[-1]]
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
