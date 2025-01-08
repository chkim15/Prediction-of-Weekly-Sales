# Walmart Sales Prediction Using Advanced ML Techniques

A comprehensive analysis comparing multiple machine learning approaches to predict weekly sales across 45 Walmart stores using economic indicators and store-specific data. This project demonstrates expertise in time series analysis, ensemble methods, and thorough model evaluation.

## Business Problem

Retail stores need accurate sales forecasting to optimize operations and inventory management. This project analyzes how different economic conditions (CPI, unemployment, fuel prices) and store-specific factors influence weekly sales, helping to:
- Predict future sales performance
- Understand seasonal patterns and holiday effects
- Identify key economic indicators affecting sales
- Enable data-driven operational planning

## Dataset

- **Scope**: 45 Walmart stores (2010-2012)
- **Records**: 6,435 entries with no missing values
- **Features**: Store number, dates, holiday flags, temperature, fuel price, CPI, unemployment rate
- **Target**: Weekly sales ($209,986 - $3,818,686)
- **Special Events**: Super Bowl, Labor Day, Thanksgiving, Christmas

## Methodology

### 1. Linear Regression (Baseline)
- Log-transformed sales for multiplicative effects
- One-hot encoded categorical variables
- Implemented 10-fold cross-validation with shuffling
- Achieved R² = 0.975 on test set
- Thoroughly validated regression assumptions

### 2. Time Series Analysis (SARIMA)
- SARIMAX(1,1,1)x(1,1,1,52) for seasonal patterns
- ARIMA(4,1,4) for trend-focused stores
- Identified distinct store patterns using ACF/PACF analysis
- Best RMSE performance: 56,730 (seasonal) / 13,146 (trend)

### 3. Random Forest
- Implemented both with/without one-hot encoding
- Optimized hyperparameters using OOB scores
- 100 estimators with 0.5 feature sampling
- R² = 0.958 on test set

### 4. XGBoost
- Achieved best overall performance
- R² = 0.978 on test set
- Used GridSearchCV for hyperparameter optimization
- Robust performance without feature engineering

## Key Findings

1. **Model Performance**
   - XGBoost achieved highest accuracy (R² = 0.978)
   - SARIMA showed superior short-term forecasting
   - Linear regression provided strong baseline (R² = 0.975)

2. **Feature Importance**
   - Store location was the dominant predictor
   - Unemployment rate showed significant impact
   - Holiday periods demonstrated clear sales patterns

3. **Technical Insights**
   - Store-specific trends require specialized modeling
   - Seasonal patterns strongly influence predictions
   - Economic indicators have varying impact by location

## Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Statsmodels
- Matplotlib, Seaborn

## Future Improvements

- Integration of additional economic indicators
- Deep learning implementation
- Incorporation of competitor data
- More sophisticated seasonal pattern analysis
- Extension to more recent data
