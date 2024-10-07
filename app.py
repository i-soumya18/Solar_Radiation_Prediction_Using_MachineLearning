import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For loading the saved model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import plotly.express as px  # For interactive plots

# Load your model and dataset
model = joblib.load('xgboost_model.pkl')  # Load your trained model
df = pd.read_csv('Solar_Prediction.csv')  # Load your dataset

# App title
st.title("Rooftop Solar Energy Prediction & Insights")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

# Create sliders or input fields for user input
temperature = st.sidebar.slider("Temperature (°C)", -10, 80, 25)
pressure = st.sidebar.slider("Pressure (hPa)", 900, 1100, 1000)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 50)
windDirection = st.sidebar.slider("Wind Direction (Degrees)", 0, 360, 180)
speed = st.sidebar.slider("Speed (m/s)", 0, 20, 5)

# Create a DataFrame for user input
input_data = pd.DataFrame({
    'Temperature': [temperature],
    'Pressure': [pressure],
    'Humidity': [humidity],
    'WindDirection(Degrees)': [windDirection],
    'Speed': [speed]
})

# Make predictions
prediction = model.predict(input_data)

# Display prediction
st.subheader("Prediction")
st.write(f"Predicted Solar Energy Output: {prediction[0]:.2f} W")

# Statistics
st.subheader("Model Statistics")
try:
    # Check if the target column exists
    if 'Radiation' in df.columns:
        y_true = df['Radiation']
        y_pred = model.predict(df.drop('Radiation', axis=1))

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Cross-validation
        cv_mse = cross_val_score(model, df.drop('Radiation', axis=1), df['Radiation'], cv=5, scoring='neg_mean_squared_error')
        cv_mse_mean = -cv_mse.mean()

        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"R² Score: {r2:.2f}")
        st.write(f"Cross-Validation MSE: {cv_mse_mean:.2f}")
    else:
        st.error("Error: 'Radiation' column not found in the dataset.")
except Exception as e:
    st.error(f"An error occurred: {e}")

# Feature Importance Visualization
st.subheader("Feature Importance")
feature_importance = model.feature_importances_  # XGBoost model attribute
feature_names = df.drop('Radiation', axis=1).columns  # Features without the target variable

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Plotting feature importance
fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importance',
             labels={'Importance': 'Importance Score', 'Feature': 'Features'})
st.plotly_chart(fig)

# Additional Visualizations
st.subheader("Data Insights and Visualizations")

# Correlation Heatmap
st.subheader("Correlation Heatmap")
corr_matrix = df.corr()
fig, ax = plt.subplots()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Pairplot to visualize relationships
st.subheader("Pairplot of Features")
pairplot_fig = sns.pairplot(df)
st.pyplot(pairplot_fig)

# Interactive Line Plot for Solar Energy Output with respect to Temperature
st.subheader("Solar Energy Output vs Temperature")
fig_line = px.line(df, x='Temperature', y='Radiation', title="Solar Energy Output vs Temperature",
                   labels={'Temperature': 'Temperature (°C)', 'Radiation': 'Solar Energy Output (W)'})
st.plotly_chart(fig_line)

# Histograms for individual features
st.subheader("Feature Distribution")
for col in df.columns:
    fig_hist = px.histogram(df, x=col, title=f'{col} Distribution')
    st.plotly_chart(fig_hist)


# Actual vs Predicted scatter plot
st.subheader("Actual vs Predicted")
fig, ax = plt.subplots()
sns.scatterplot(x=y_true, y=y_pred, ax=ax)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
st.pyplot(fig)
