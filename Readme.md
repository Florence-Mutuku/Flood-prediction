
# Capstone Project: Flood Probability Prediction

## Overview

This project develops machine learning models to predict flood probabilities, enhancing flood risk assessment for improved mitigation, forecasting, and management. Accurate predictions are crucial for minimizing flood-related damage. The analysis uses the "Flood Prediction" dataset by Naiya Khalid (Kaggle: [https://www.kaggle.com/datasets/naiyakhaled/flood-prediction](https://www.kaggle.com/datasets/naiyakhaled/flood-prediction)).

## Dataset

* **Source:** Kaggle (Naiya Khalid)
* **Shape:** (50000, 21)
* **Features:** 20 (capturing weather, land, infrastructure, human, and environmental factors)
* **Target:** `FloodProbability` (continuous, 0.285 to 0.725)

**Key Feature Categories:**

* Weather (MonsoonIntensity)
* Land & Water (TopographyDrainage, RiverManagement, Watersheds)
* Infrastructure & Human (DamsQuality, DrainageSystems, Urbanization, PopulationScore, InadequatePlanning)
* Environment (Deforestation, WetlandLoss, Siltation, CoastalVulnerability, Landslides)
* Other (AgriculturalPractices, Encroachments, IneffectiveDisasterPreparedness, PoliticalFactors, ClimateChange, DeterioratingInfrastructure)

## Implementation

**Models:**

* Linear Regression
* Random Forest
* Gradient Boosting
* XGBoost

**Preprocessing:**

* Missing value handling: No missing values found.
* Feature scaling: MinMaxScaler
* Train-test split: 80/20

**Evaluation Metrics:**

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² (R-squared)

## Results

**Model Performance Comparison**

| Model             | RMSE   | MAE    | R²     |
|-------------------|--------|--------|--------|
| Linear Regression | 0.0207 | 0.0166 | 0.8206 |
| Random Forest     | 0.0216 | 0.0174 | 0.8133 |
| Gradient Boosting | 0.0211 | 0.0170 | 0.8173 |
| XGBoost           | 0.0212 | 0.0171 | 0.8160 |

* **Best Model:** Linear Regression

## Visualizations (in Notebook)

* Correlation Matrix (feature dependencies)
* Feature Importance Plot (Random Forest, Gradient Boosting, XGBoost)
* Actual vs. Predicted Values Scatter Plot
* Residual Analysis
## Project Structure
flood-prediction/
│
├── data/
│   └── flood.csv                  # Main dataset
│
├── notebooks/
│   └── flood_prediction_notebook.ipynb   # Analysis notebook
│
├── models/
│   ├── linear_regression.pkl      # Saved Linear Regression model
│   ├── random_forest.pkl          # Saved Random Forest model  
│   ├── gradient_boosting.pkl      # Saved Gradient Boosting model
│   └── xgboost.pkl                # Saved XGBoost model
│
├── requirements.txt               # Dependencies
└── README.md                      # This file

## Setup

1.  Python 3.x
2.  `pip install pandas numpy scikit-learn matplotlib seaborn statsmodels xgboost shap joblib tensorflow`
3.  `flood.csv` in the same directory as the notebook.
4.  Run `flood prediction notebook.ipynb` in Jupyter.

## Future Work
Several avenues exist for future research and development to enhance the flood prediction capabilities of this project:

* Address potential data leakage
* Incorporate LSTM or other time-series models
* Incorporate Temporal Data: Future work will explore the incorporation of time-series data, such as historical rainfall patterns, river flow rates, and seasonal variations, to capture the temporal dynamics of flood events.
* Integrate Remote Sensing Data: The integration of remote sensing data, such as satellite imagery and radar data, could provide valuable real-time information on land use, water levels, and flood extent, enabling more accurate and timely predictions.
*  Develop a User-Friendly Interface: A user-friendly interface could be developed to make the flood prediction models accessible to a wider audience, including government agencies, disaster management organizations, and the general public. This interface could provide interactive maps, risk assessments, and early warning alerts.

## Author

Florence Ndungwa
https://github.com/Florence-Mutuku
https://www.linkedin.com/in/florencendungwa

## License

MIT License

Copyright (c) 2025 Florence Ndungwa

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.



## Installation
1.Clone the repository
  bash
  git clone https://github.com/Florence-Mutuku/flood-prediction.git
  cd flood-prediction

2.Create and activate a virtual environment (optional but recommended)
 bash
 python -m venv env
 source env/bin/activate  # On Windows: env\Scripts\activate

3.Install dependencies
  bash
  pip install -r requirements.txt

4.Ensure flood.csv is in the data directory or update the path in the notebook
## Usage
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler  # Import MinMaxScaler

# Load the trained model
model = joblib.load('models/linear_regression.pkl')

# Load the scaler (IMPORTANT: Scale input data!)
scaler = joblib.load('models/flood_prediction_scaler.pkl') # Assuming you saved the scaler as this

# Prepare input data (example - ALL 20 features are REQUIRED, in the correct order)
input_data = {
    'MonsoonIntensity': 0.65,
    'TopographyDrainage': 0.58,
    'RiverManagement': 0.45,
    'DamsQuality': 0.72,
    'DrainageSystems': 0.61,
    'Urbanization': 0.52,
    'PopulationScore': 0.49,
    'InadequatePlanning': 0.68,
    'Deforestation': 0.55,
    'WetlandLoss': 0.41,
    'Siltation': 0.70,
    'CoastalVulnerability': 0.38,
    'Landslides': 0.63,
    'AgriculturalPractices': 0.59,
    'Encroachments': 0.66,
    'IneffectiveDisasterPreparedness': 0.75,
    'PoliticalFactors': 0.47,
    'ClimateChange': 0.50,
    'DeterioratingInfrastructure': 0.62,
    'Watersheds': 0.53  # Add the missing feature
}

# Convert to DataFrame
df = pd.DataFrame([input_data])

# Scale the input data using the loaded scaler (CRUCIAL STEP)
df_scaled = scaler.transform(df)

# Make prediction
try:
    flood_probability = model.predict(df_scaled)[0]
    print(f"Predicted flood probability: {flood_probability:.4f}")
except ValueError as e:
    print(f"Error: {e}. Please ensure all 20 features are provided in the correct order and format.")

