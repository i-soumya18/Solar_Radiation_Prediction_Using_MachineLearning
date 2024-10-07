import pandas as pd
df = pd.read_csv('Solar_Prediction.csv')

from sklearn.ensemble import RandomForestRegressor

# Assuming you've prepared your features and target
X = df[['Temperature', 'Humidity', 'Pressure', 'Speed']]  # Add other features as needed
y = df['Radiation']

model = RandomForestRegressor()
model.fit(X, y)

# Get feature importance
importance = model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})

# Sort by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print(importance_df)
