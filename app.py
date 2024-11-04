# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import numpy as np

# Load data with error handling
try:
    df = pd.read_csv('Solar_Prediction.csv')
    st.success("Data loaded successfully!")
except FileNotFoundError:
    st.error("Data file not found. Please ensure 'Solar_Prediction.csv' is available.")
    st.stop()

# Title and introduction
st.title("Solar Radiation Prediction and Model Comparison")
st.write("Analyze, predict, and compare different models for solar radiation data.")

# Data preview and statistics with error handling
try:
    if st.checkbox("Show data preview"):
        st.write(df.head())

    if st.checkbox("Show data statistics"):
        st.write(df.describe())
except Exception as e:
    st.error(f"Error displaying data: {e}")

# Check for missing values and display notification
missing_values = df.isnull().sum().sum()
if missing_values > 0:
    st.warning(f"Data contains {missing_values} missing values. Handle missing values before training models.")
else:
    st.success("No missing values in the dataset.")

# Model Training and Evaluation
st.subheader("Model Training and Evaluation")

try:
    # Splitting data
    X = df.drop('Radiation', axis=1)  # Assuming 'Radiation' is the target
    y = df['Radiation']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
except KeyError:
    st.error("Column 'Radiation' not found in data. Check if your dataset contains this column.")
    st.stop()

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'XGBoost': XGBRegressor()
}

# Dictionary to store model performance
model_metrics = {}

# Train models and compute metrics with error handling
for name, model in models.items():
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        # Store metrics in dictionary
        model_metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R^2': r2}
    except Exception as e:
        st.error(f"Error training {name}: {e}")

# Model selection for prediction
st.subheader("Select a Model for Prediction")

selected_model_name = st.selectbox("Choose a model:", list(models.keys()))
selected_model = models[selected_model_name]

# Display model metrics with error handling
try:
    st.write(f"**Performance Metrics for {selected_model_name}:**")
    st.write("MAE:", model_metrics[selected_model_name]['MAE'])
    st.write("RMSE:", model_metrics[selected_model_name]['RMSE'])
    st.write("R²:", model_metrics[selected_model_name]['R^2'])
except KeyError:
    st.error("Model metrics not available. Ensure the model has been trained properly.")

# Real-time Prediction
st.subheader("Real-Time Prediction")

# User input for prediction with validation
try:
    temp = st.number_input("Temperature")
    humidity = st.number_input("Humidity")
    pressure = st.number_input("Pressure")
    speed = st.number_input("Speed")
    wind_direction = st.number_input("Wind Direction (Degrees)")

    if st.button("Predict Radiation"):
        input_data = [[temp, humidity, pressure, speed, wind_direction]]
        prediction = selected_model.predict(input_data)
        st.write(f"Predicted Radiation: {prediction[0]:.2f}")
except Exception as e:
    st.error(f"Prediction failed: {e}")

# Model Comparison Table and Bar Chart
st.subheader("Model Comparison Table and Visualization")

# Convert model_metrics to DataFrame for display
try:
    metrics_df = pd.DataFrame(model_metrics).T.reset_index().rename(columns={'index': 'Model'})
    st.write(metrics_df)

    # Display comparison bar chart
    fig = px.bar(metrics_df, x='Model', y=['MAE', 'RMSE', 'R^2'],
                 title="Model Performance Comparison",
                 labels={'value': 'Score', 'Model': 'Model'},
                 barmode='group')
    st.plotly_chart(fig)
except Exception as e:
    st.error(f"Error displaying model metrics: {e}")

# Optional data visualization - Distribution of features
if st.checkbox("Show feature distributions"):
    st.subheader("Feature Distributions")
    features = ['Radiation', 'Temperature', 'Humidity', 'Pressure', 'Speed', 'WindDirection(Degrees)']
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']  # Colors for each feature

    for feature, color in zip(features, colors):
        try:
            fig = px.histogram(df, x=feature, color_discrete_sequence=[color],
                               title=f'Distribution of {feature}',
                               marginal='box', opacity=0.7)
            fig.update_layout(xaxis_title=feature, yaxis_title='Count')
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error displaying distribution for {feature}: {e}")

# Correlation Heatmap
if st.checkbox("Show correlation heatmap"):
    try:
        st.subheader("Feature Correlation Heatmap")
        fig = px.imshow(df.corr(), text_auto=True, color_continuous_scale='Viridis')
        st.plotly_chart(fig)
    except Exception as e:
        st.error(f"Error displaying correlation heatmap: {e}")

# Interactive Scatter Plots
if st.checkbox("Show scatter plots"):
    st.subheader("Scatter Plots of Features vs Radiation")
    for feature in ['Temperature', 'Humidity', 'Pressure', 'Speed', 'WindDirection(Degrees)']:
        try:
            fig = px.scatter(df, x=feature, y='Radiation', title=f'{feature} vs Radiation')
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error displaying scatter plot for {feature}: {e}")
