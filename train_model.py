import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import joblib
import numpy as np
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- Step 1: Load the Dataset ---
try:
    data = pd.read_csv('youtube_channel_real_performance_analytics.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'youtube_channel_real_performance_analytics.csv' not found.")
    print("Please make sure the CSV file is in the same folder as this script.")
    exit()

# --- Step 2: Data Cleaning and Preprocessing ---
print("Starting data cleaning...")
# Fill NaNs and handle object types
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = pd.to_numeric(data[col], errors='coerce')
    if data[col].isnull().sum() > 0:
        data[col] = data[col].fillna(data[col].median())

# Ensure revenue is non-negative and drop any remaining NaNs
data.dropna(subset=['Estimated Revenue (USD)'], inplace=True)
data = data[data['Estimated Revenue (USD)'] >= 0]
print("Data cleaning complete. Shape:", data.shape)


# --- Step 3: Feature Engineering and Selection ---
print("Performing feature engineering...")
# Create a more robust engagement metric
epsilon = 1e-6 
data['EngagementRate'] = (data['New Comments'] + data['Likes'] + data['Shares']) / (data['Views'] + epsilon)

# Define the feature set
features = [
    'Views', 'Likes', 'New Comments', 'Shares', 'Subscribers', 'Video Duration', 
    'Average View Percentage (%)', 'Ad Impressions', 'Watch Time (hours)',
    'Unique Viewers', 'EngagementRate'
]
target = 'Estimated Revenue (USD)'

X = data[features]
y = data[target]

# Apply a log transformation to the target variable to handle skewed data
y_log = np.log1p(y)

# Split the data
X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)
print(f"Data split. Training set: {X_train.shape[0]}, Testing set: {X_test.shape[0]}")


# --- Step 4: Hyperparameter Tuning ---
print("\nStarting hyperparameter tuning... (This may take several minutes)")

# Define the parameter grid for RandomForest
param_grid_rf = {
    'n_estimators': [100, 200], 'max_depth': [10, 20, None],
    'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]
}

# Define the parameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5, 7], 'subsample': [0.8, 1.0]
}

# Tune RandomForest using GridSearchCV
grid_search_rf = GridSearchCV(estimator=RandomForestRegressor(random_state=42), 
                              param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=0)
grid_search_rf.fit(X_train, y_train_log)
best_rf = grid_search_rf.best_estimator_
print("Best RandomForest parameters found.")

# Tune XGBoost using GridSearchCV
grid_search_xgb = GridSearchCV(estimator=xgb.XGBRegressor(objective='reg:squarederror', random_state=42),
                               param_grid=param_grid_xgb, cv=3, n_jobs=-1, verbose=0)
grid_search_xgb.fit(X_train, y_train_log)
best_xgb = grid_search_xgb.best_estimator_
print("Best XGBoost parameters found.")


# --- Step 5: Train Final Model ---
# Create the ensemble model using the best-tuned estimators
ensemble_model = VotingRegressor(estimators=[('rf', best_rf), ('xgb', best_xgb)])

print("\nTraining the final tuned Ensemble model...")
ensemble_model.fit(X_train, y_train_log)
print("Model training complete.")


# --- Step 6: Evaluate and Save ---
# Evaluate the model
y_pred_log = ensemble_model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test = np.expm1(y_test_log)
y_pred[y_pred < 0] = 0
r2 = r2_score(y_test, y_pred)
print(f"\n--- Final Model Performance ---")
print(f"R-squared (R2 Score): {r2:.2f}")
print("-----------------------------\n")

# Save the final model and features list
# This will create the files your Flask app needs
joblib.dump(ensemble_model, 'model.pkl')
joblib.dump(features, 'features.pkl')
print("Final model saved as 'model.pkl'")
print("Feature list saved as 'features.pkl'")
print("You can now run your Flask app.")
