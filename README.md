# Rooftop Solar Energy Prediction

## Overview

This project aims to predict solar energy output using various meteorological parameters. It leverages machine learning models and interactive visualization tools to provide accurate predictions of rooftop solar energy generation. The application is built using Streamlit for a smooth UI/UX and Plotly for rich, interactive data visualizations.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Modeling Approach](#modeling-approach)
- [Visualizations](#visualizations)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Live Demo](#live-demo)
- [Contributing](#contributing)
- [License](#license)

## Features

- Interactive UI for predicting solar energy output based on user-defined inputs like temperature, humidity, wind speed, etc.
- Machine Learning Models for accurate predictions using historical data.
- Visualization Dashboard displaying model performance metrics and feature importance.
- Correlation Heatmaps to explore feature relationships.
- Outlier Detection using box plots.
- Actual vs Predicted energy output comparisons.
- Seasonal Decomposition to visualize patterns in energy production and weather parameters.

## Modeling Approach

This project involves training and evaluating multiple machine learning models to predict solar radiation output based on meteorological features:

### Models Used:
- XGBoost
- Gradient Boosting
- Random Forest
- Decision Tree
- Linear Regression

### Model Evaluation Metrics:
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- R² Score

The best-performing model is then deployed in the Streamlit app for real-time predictions.

## Visualizations

We employ Plotly to create interactive visualizations, such as:

- **Feature Distributions**: Histograms and box plots for features like temperature, humidity, and pressure.
- **Correlation Analysis**: Heatmaps and bar charts showing feature correlations with solar radiation.
- **Scatter Plots**: Comparing actual vs predicted values for radiation.
- **Seasonal Decomposition**: Visualizing the trend, seasonality, and residuals for energy output and weather parameters.

## Technology Stack

- **Backend**: Python, Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn, XGBoost
- **Visualization**: Plotly, Seaborn, Matplotlib
- **Deployment**: Streamlit Cloud, GitHub

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/i-soumya18/Solar_Radiation_Prediction_Using_MachineLearning.git
    cd Solar_Radiation_Prediction
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

4. **Model Training (Optional)**: Train the machine learning models using the provided notebook or scripts before deploying the app.

## Usage

Once the application is running, you can use the Streamlit interface to:

- Input various meteorological parameters (temperature, pressure, humidity, etc.)
- Get solar energy output predictions.
- Explore visualizations for feature importance, correlation, and seasonal decomposition.

### How to Use:

- Use the side panel to adjust meteorological parameters using sliders.
- The predicted solar energy output will be displayed instantly.
- Scroll down to explore interactive charts and insights on feature importance and model performance.

## Screenshots

Here are some screenshots of the project:

- **Prediction Interface**  
  ![Prediction Interface](/![interface](https://github.com/user-attachments/assets/fa57c2d4-6a18-4a1c-b048-ee979076257b)
)

- **Feature Importance**  
  ![Feature Importance](/importance.png)

- **Correlation Heatmap**  
  ![Correlation Heatmap](/heatmap.png)

- **Seasonal Decomposition**  
  ![Seasonal Decomposition](/decomposition.png)

## Live Demo

Check out the live demo of the application [here](https://solar-radiation-prediction.streamlit.app/). The deployed app allows users to interact with the model and view visualizations of the dataset.
## Contributing

If you wish to contribute to this project:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature_branch`).
3. Commit your changes (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature_branch`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
