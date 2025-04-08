import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import joblib

np.random.seed(42)
n = 500
df = pd.DataFrame({
    'team_experience_years': np.random.randint(0, 20, n),
    'past_successes': np.random.randint(0, 5, n),
    'funding_rounds': np.random.randint(0, 10, n),
    'total_funding': np.random.uniform(0.5, 100, n),
    'has_top_investor': np.random.choice(['yes', 'no'], n),
    'debt_ratio': np.random.uniform(0.0, 1.0, n),
    'cash_flow_score': np.random.uniform(0.0, 1.0, n),
    'industry': np.random.choice(['Fintech', 'Healthtech', 'Edtech', 'E-commerce'], n),
    'regulatory_risk_score': np.random.uniform(0.0, 1.0, n),
    'customer_retention_rate': np.random.uniform(0.0, 1.0, n),
    'customer_satisfaction_score': np.random.uniform(0, 10, n),
    'survived_5_years': np.random.choice([0, 1], n)
})

X = df.drop('survived_5_years', axis=1)
y = df['survived_5_years']

numeric = [
    'team_experience_years', 'past_successes', 'funding_rounds',
    'total_funding', 'debt_ratio', 'cash_flow_score',
    'regulatory_risk_score', 'customer_retention_rate',
    'customer_satisfaction_score'
]
categorical = ['industry', 'has_top_investor']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

pipeline.fit(X, y)
joblib.dump(pipeline, 'startup_model.pkl')
print("✅ Model saved as startup_model.pkl")
