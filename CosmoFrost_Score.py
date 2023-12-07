import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# The choice of the model (Random Forest)

# This AI model would allows to input the parameters of a cooling system 
# and predict its CosmoFrost Score, aiding in decision-making for selecting 
# the most suitable cooling system for spacecraft

# Experiment Dataset
data = {
    'energy_efficiency': [70, 92, 85, 90],  # example values
    'spatial_footprint': [80, 66, 75, 78],  # example values
    'cooling_effectiveness': [75, 85, 80, 82],  # example values
    'operational_reliability': [80, 85, 78, 81],  # example values
    'cost_effectiveness': [60, 70, 65, 68],  # example values
    'cosmofrost_score': [75, 88, 82, 85]  # example target values
}

df = pd.DataFrame(data)

# Splitting data into features and target
X = df.drop('cosmofrost_score', axis=1)
y = df['cosmofrost_score']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Model prediction and evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

