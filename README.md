# Walmart Store Sales Prediction

This project develops and compares machine learning models to predict weekly sales for 45 Walmart stores based on economic indicators and store-specific data.

## Project Overview

- **Dataset**: Historical sales data from 2010-2012 covering 45 Walmart stores
- **Target Variable**: Weekly sales per store
- **Features**: Store number, date, holiday flags, temperature, fuel price, CPI, unemployment rate
- **Records**: 6,435 rows with no missing values

## Methods & Models

Implemented and compared multiple approaches:

1. **Linear Regression (Baseline)**
   - Log-transformed sales data
   - One-hot encoded categorical variables
   - Achieved R² = 0.975 on test set

2. **SARIMA Time Series Analysis**
   - SARIMAX(1,1,1)x(1,1,1,52) for seasonal stores
   - ARIMA(4,1,4) for trend stores
   - Best RMSE performance among all models

3. **Random Forest**
   - Handled categorical variables without encoding
   - R² = 0.958 on test set
   - Feature importance analysis showed store number as key predictor

4. **XGBoost**
   - Best performing ML model
   - R² = 0.978 on test set
   - Excellent performance without feature engineering

## Key Findings

- SARIMA models showed superior performance for short-term predictions
- XGBoost achieved best overall R² score among ML models
- Store number and unemployment rate were top predictive features
- Holiday periods showed significant sales variations

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Statsmodels (SARIMA)

## Future Work

- Integration of additional economic indicators
- Exploration of deep learning approaches
- Incorporation of more recent data
- Investigation of store-specific patterns
