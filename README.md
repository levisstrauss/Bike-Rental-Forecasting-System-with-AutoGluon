# 🚲 Bike Sharing Demand Prediction with AutoGluon

## 📋 Project Overview

This project tackles the Kaggle Bike Sharing Demand competition using AutoGluon's automated machine learning capabilities. Through iterative model 
development and feature engineering, I achieved significant performance improvements, reaching a final RMSE score of 0.45061 - a 66% improvement over the baseline model.

The project demonstrates a systematic approach to solving a real-world regression problem:

  - Implementation of automated machine learning workflows
  - Feature engineering with temporal data
  - Hyperparameter optimization for model performance
  - Iterative model improvement and evaluation

## 🔍 Data Analysis & Insights

The project utilizes a dataset of hourly bike rentals with historical weather and seasonal information.
Key patterns revealed through exploratory data analysis include:
<div>
  <img src="https://github.com/levisstrauss/Bike-Rental-Forecasting-System-with-AutoGluon/blob/main/project/img/sharing.png" alt="Hourly Bike Rental Patterns" width="70%">
</div>

The analysis revealed strong temporal patterns in bike rental behavior:

- Peak Demand: 8 AM and 5-6 PM show highest rental volumes, aligned with commuting hours
- Weekend vs. Weekday: Different demand profiles for working days versus weekends
- Seasonal Trends: Significant variation between summer and winter months

## Feature Correlations:

- Temperature Impact: Strong positive correlation (0.63) between temperature and rental count
- Weather Conditions: Negative correlation (-0.30) between humidity and rentals
- Seasonal Effects: Clear seasonal patterns reflected in the season-count relationship

## 🛠️ Methodology & Implementation 

```python
# Create datetime-based features
train['hour'] = train['datetime'].dt.hour
train['day'] = train['datetime'].dt.day
train['month'] = train['datetime'].dt.month
train['dayofweek'] = train['datetime'].dt.dayofweek
train['year'] = train['datetime'].dt.year

# Convert categorical features
train["season"] = train["season"].astype('category')
train["weather"] = train["weather"].astype('category')
train["hour"] = train["hour"].astype('category')
```
## Model Development:

The project followed an iterative development approach with three key stages:

1. Baseline Model:

```python
predictor = TabularPredictor(
    label='count',
    eval_metric='root_mean_squared_error',
    path='models/ag_models'
).fit(
    train_data=train.drop(exclude_columns, axis=1),
    time_limit=600,
    presets='best_quality'
)
```
2. Feature-Enhanced Model:
   
Built upon the baseline by adding engineered temporal features from the datetime column

3. Hyperparameter-Optimized Model:

```bash
hyperparameters = {
    'GBM': [
        {},
        {'extra_trees': True, 'ag_args': {'name_suffix': 'XT'}},
        {
            'learning_rate': 0.01,
            'num_leaves': 128,
            'feature_fraction': 0.8,
            'min_data_in_leaf': 5,
            'num_boost_round': 500,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'ag_args': {'name_suffix': 'Tuned'}
        }
    ],
    'CAT': [
        {},
        {
            'depth': 8,
            'learning_rate': 0.02,
            'l2_leaf_reg': 3.0,
            'grow_policy': 'Depthwise',
            'iterations': 500,
            'ag_args': {'name_suffix': 'Tuned'}
        }
    ],
    'RF': [
        {'criterion': 'squared_error', 'max_depth': 15, 'n_estimators': 300}
    ],
    'XT': [
        {'criterion': 'squared_error', 'max_depth': 15, 'n_estimators': 300}
    ]
}
```
## 📊 Results & Improvement Path

| Model Stage           | RMSE Score | Improvement | Key Changes                                             |
| --------------------- | ---------- | ----------- | ------------------------------------------------------- |
| Initial Baseline      | 1.32218    | –           | Base AutoGluon model                                    |
| Feature Engineering   | 0.47449    | 64.1%       | Added features: hour, day, month, dayofweek             |
| Hyperparameter Tuning | 0.45061    | 5.0%        | Tuned `learning_rate=0.01`, `num_leaves=128`, `depth=8` |

## Key Findings:

1. Feature Engineering Impact: Creating temporal features from the datetime column provided the most substantial performance improvement (64.1%)
   
2. Hyperparameter Optimization: Further tuning delivered additional gains, with gradient boosting parameters showing the most influence:

  - Reduced learning rate (0.01) for more conservative, stable learning
  - Increased tree complexity (num_leaves: 128) for capturing more complex patterns
  - Optimal tree depth (8) balancing detail capture and generalization

3. Model Comparison:
 AutoGluon's ensemble approach effectively combined multiple algorithms, with gradient boosting models (GBM) providing the strongest performance

## 🧠 Insights & Lessons Learned

### Technical Insights:

  1. Feature Engineering Priority:
  Well-crafted features based on domain knowledge provided significantly greater improvement than algorithmic tuning alone
  
  2. Temporal Data Handling:
  Converting cyclic time features (hour, day, month) to categorical variables helped the model capture periodic patterns
  
  3. AutoGluon Capabilities: The framework effectively handled:

  - Complex feature relationships
  - Automated model selection and ensembling
  - Multi-algorithm stacking and blending
  
## Development Insights:

  1. Iterative Improvement: The systematic approach of baseline → feature engineering → hyperparameter tuning provided clear performance gains at each stage
  2. Computation/Performance Trade-offs: Extending training time from 10 to 10+ minutes allowed for more complex models but with diminishing returns
  3. Model Interpretability: While ensemble models provided the best performance, they reduced interpretability compared to single models

## 🚀 Usage & Reproduction

```bash
Environment Setup:
bash# Update pip and essential packages
pip install -U pip setuptools wheel

# Install required packages
pip install "bokeh>=2.4.3,<3"
pip install "mxnet<2.0.0"
pip install autogluon --no-cache-dir
```

## Running the Project:

```python
python# Load the data
train = pd.read_csv('train.csv', parse_dates=['datetime'])
test = pd.read_csv('test.csv', parse_dates=['datetime'])

# Create features and train model
# [Feature engineering code here]

# Train with AutoGluon
predictor = TabularPredictor(
    label='count',
    eval_metric='root_mean_squared_error'
).fit(
    train_data=train_processed,
    time_limit=600,
    presets='best_quality'
)

# Generate predictions
predictions = predictor.predict(test_processed)
```
### 🔮 Future Improvements

Future work could explore several promising directions:

1. Advanced Feature Engineering:

  - Weather interaction features (temp × humidity, weather × hour)
  - More sophisticated temporal features (holiday proximity, weekend vs. weekday patterns)
  - External data integration (events, public transportation schedules)

2. Enhanced Hyperparameter Optimization:

  - More extensive hyperparameter search with increased computational resources
  - Targeted optimization of stacking and blending parameters
  - Model-specific parameter tuning for the strongest performers
  
3. Alternative Approaches:

  - Dedicated time series models for temporal dependencies
  - Deep learning approaches for complex pattern detection
  - Separate models for different time periods or weather conditions

## 📚 Key Takeaways

This project demonstrates the effectiveness of combining automated machine learning with thoughtful feature engineering. While AutoGluon provides strong out-of-box performance, domain knowledge applied to feature creation yields the most substantial improvements.

The progression from a baseline RMSE of 1.32218 to a final score of 0.45061 highlights the value of an iterative, systematic approach to machine learning development. Each enhancement step provided measurable performance gains, with feature engineering delivering the most significant impact - reinforcing that good features often matter more than algorithm selection or hyperparameter tuning.

## 🙏 Acknowledgments

  - Kaggle for hosting the Bike Sharing Demand competition
  - AWS and Udacity for the project framework and computational resources (Sagemaker)
  - AutoGluon team for developing this powerful automated ML library






























