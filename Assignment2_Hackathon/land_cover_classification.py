#!/usr/bin/env python3

# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def get_ndvi_columns(df):
    """Get NDVI columns from the dataframe."""
    return [col for col in df.columns if '_N' in col]

def preprocess_data(df, is_train=True):
    """Preprocess the data by handling missing values and engineering features."""
    
    # Get NDVI columns
    ndvi_cols = get_ndvi_columns(df)
    
    # Separate features and target
    if is_train:
        X = df[ndvi_cols].copy()
        y = df['class']
    else:
        X = df[ndvi_cols].copy()
        y = None
    
    # 1. Handle missing values using median imputation
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(
        imputer.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # 2. Calculate rolling statistics (3-point window)
    rolling_stats = X_imputed.rolling(window=3, min_periods=1, axis=1)
    X_rolling_mean = rolling_stats.mean().fillna(method='ffill').fillna(method='bfill')
    X_rolling_std = rolling_stats.std().fillna(method='ffill').fillna(method='bfill')
    
    # Name the rolling statistics columns
    X_rolling_mean.columns = [f'{col}_rolling_mean' for col in X_rolling_mean.columns]
    X_rolling_std.columns = [f'{col}_rolling_std' for col in X_rolling_std.columns]
    
    # 3. Calculate temporal differences
    X_diff = X_imputed.diff(axis=1).fillna(0)
    X_diff.columns = [f'{col}_diff' for col in X_diff.columns]
    
    # 4. Calculate global statistics
    global_stats = pd.DataFrame(index=X_imputed.index)
    global_stats['max_ndvi'] = X_imputed.max(axis=1)
    global_stats['min_ndvi'] = X_imputed.min(axis=1)
    global_stats['range_ndvi'] = global_stats['max_ndvi'] - global_stats['min_ndvi']
    global_stats['mean_ndvi'] = X_imputed.mean(axis=1)
    global_stats['std_ndvi'] = X_imputed.std(axis=1)
    
    # 5. Calculate seasonal features (assuming dates are in chronological order)
    n_seasons = 4
    season_size = X_imputed.shape[1] // n_seasons
    seasonal_stats = pd.DataFrame(index=X_imputed.index)
    
    for i in range(n_seasons):
        start_idx = i * season_size
        end_idx = (i + 1) * season_size if i < n_seasons - 1 else X_imputed.shape[1]
        season_data = X_imputed.iloc[:, start_idx:end_idx]
        seasonal_stats[f'season_{i+1}_mean'] = season_data.mean(axis=1)
        seasonal_stats[f'season_{i+1}_std'] = season_data.std(axis=1)
    
    # Combine all features
    features = pd.concat([
        X_imputed,        # Original imputed NDVI values
        X_rolling_mean,   # Rolling means
        X_rolling_std,    # Rolling standard deviations
        X_diff,           # Temporal differences
        seasonal_stats,   # Seasonal statistics
        global_stats      # Global statistics
    ], axis=1)
    
    # Final check for any remaining NaN values
    if features.isna().any().any():
        features = features.fillna(features.mean())
    
    return features, y

def main():
    # Load the datasets
    train_df = pd.read_csv('hacktrain.csv')
    test_df = pd.read_csv('hacktest.csv')

    print("Training data shape:", train_df.shape)
    print("Test data shape:", test_df.shape)
    print("\nSample of training data:")
    print(train_df.head())

    # Preprocess training data
    X_train, y_train = preprocess_data(train_df, is_train=True)
    
    # Ensure no NaN values in features
    if X_train.isna().any().any():
        print("Filling remaining NaN values in training data...")
        X_train = X_train.fillna(X_train.mean())
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    
    # Ensure no NaN values after scaling
    if X_train_scaled.isna().any().any():
        print("Filling any NaN values after scaling...")
        X_train_scaled = X_train_scaled.fillna(0)

    # Train Logistic Regression model
    model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    model.fit(X_train_scaled, y_train)

    # Make predictions on training data
    train_pred = model.predict(X_train_scaled)
    print("\nTraining Accuracy:", accuracy_score(y_train, train_pred))
    print("\nClassification Report:")
    print(classification_report(y_train, train_pred))

    # Preprocess and predict on test data
    X_test, _ = preprocess_data(test_df, is_train=False)
    
    # Ensure no NaN values in test features
    if X_test.isna().any().any():
        print("Filling remaining NaN values in test data...")
        X_test = X_test.fillna(X_test.mean())
    
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Ensure no NaN values after scaling test data
    if X_test_scaled.isna().any().any():
        print("Filling any NaN values after scaling test data...")
        X_test_scaled = X_test_scaled.fillna(0)
    
    test_pred = model.predict(X_test_scaled)

    # Create submission file
    submission = pd.DataFrame({
        'ID': test_df['ID'],
        'class': test_pred
    })
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file created successfully!")
    print(submission.head())

if __name__ == "__main__":
    main() 