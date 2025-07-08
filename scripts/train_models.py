import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess the manufacturing dataset
def load_and_preprocess_data():
    """Load manufacturing data and prepare features"""
    # Simulate loading the actual dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data based on manufacturing.csv schema
    data = {
        'Temperature': np.random.normal(165, 15, n_samples),
        'Pressure': np.random.normal(25, 5, n_samples),
        'Material_Type': np.random.choice(['Steel', 'Aluminum', 'Plastic', 'Composite'], n_samples),
        'Process_Time': np.random.normal(45, 10, n_samples),
        'Humidity': np.random.normal(50, 10, n_samples),
        'Operator_Experience': np.random.choice(['Novice', 'Intermediate', 'Experienced', 'Expert'], n_samples),
        'Maintenance_Status': np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'], n_samples),
        'Batch_Size': np.random.normal(150, 50, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived features
    df['Temperature_x_Pressure'] = df['Temperature'] * df['Pressure']
    df['Material_Fusion_Metric'] = df['Temperature'] * df['Pressure'] * 0.8 + np.random.normal(0, 100, n_samples)
    df['Material_Transformation_Metric'] = (df['Temperature'] ** 2) * df['Pressure'] + np.random.normal(0, 10000, n_samples)
    
    # Generate target variable (Quality Rating)
    quality_base = (
        (df['Temperature'] - 165) * -0.1 +
        (df['Pressure'] - 25) * -0.2 +
        df['Process_Time'] * 0.05 +
        (50 - abs(df['Humidity'] - 50)) * 0.1
    )
    
    # Add material type effect
    material_effect = df['Material_Type'].map({
        'Steel': 5, 'Aluminum': 3, 'Composite': 8, 'Plastic': 2
    })
    
    # Add operator experience effect
    operator_effect = df['Operator_Experience'].map({
        'Novice': -5, 'Intermediate': 0, 'Experienced': 3, 'Expert': 7
    })
    
    # Add maintenance effect
    maintenance_effect = df['Maintenance_Status'].map({
        'Poor': -8, 'Fair': -2, 'Good': 2, 'Excellent': 6
    })
    
    df['Quality_Rating'] = (
        70 + quality_base + material_effect + operator_effect + maintenance_effect +
        np.random.normal(0, 3, n_samples)
    )
    
    # Ensure quality rating is between 0-100
    df['Quality_Rating'] = np.clip(df['Quality_Rating'], 0, 100)
    
    return df

def prepare_features(df):
    """Prepare features for machine learning models"""
    # Encode categorical variables
    le_material = LabelEncoder()
    le_operator = LabelEncoder()
    le_maintenance = LabelEncoder()
    
    df_processed = df.copy()
    df_processed['Material_Type_Encoded'] = le_material.fit_transform(df['Material_Type'])
    df_processed['Operator_Experience_Encoded'] = le_operator.fit_transform(df['Operator_Experience'])
    df_processed['Maintenance_Status_Encoded'] = le_maintenance.fit_transform(df['Maintenance_Status'])
    
    # Select features for modeling
    feature_columns = [
        'Temperature', 'Pressure', 'Temperature_x_Pressure',
        'Material_Fusion_Metric', 'Material_Transformation_Metric',
        'Process_Time', 'Humidity', 'Batch_Size',
        'Material_Type_Encoded', 'Operator_Experience_Encoded', 'Maintenance_Status_Encoded'
    ]
    
    X = df_processed[feature_columns]
    y = df_processed['Quality_Rating']
    
    return X, y, le_material, le_operator, le_maintenance

def train_mlp_model(X_train, y_train, X_test, y_test):
    """Train Multi-Layer Perceptron model"""
    print("Training MLP Neural Network...")
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'hidden_layer_sizes': [(100,), (100, 50), (150, 100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [500, 1000]
    }
    
    mlp = MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        mlp, param_grid, cv=5, scoring='neg_mean_squared_error', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_mlp = grid_search.best_estimator_
    
    # Predictions
    y_pred_train = best_mlp.predict(X_train)
    y_pred_test = best_mlp.predict(X_test)
    
    # Metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"MLP Best Parameters: {grid_search.best_params_}")
    print(f"MLP Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"MLP Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    print(f"MLP Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    return best_mlp, {
        'train_mse': train_mse, 'test_mse': test_mse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2,
        'best_params': grid_search.best_params_
    }

def train_svm_model(X_train, y_train, X_test, y_test):
    """Train Support Vector Machine with RBF kernel"""
    print("Training SVM with RBF Kernel...")
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.2]
    }
    
    svm = SVR(kernel='rbf')
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        svm, param_grid, cv=5, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_svm = grid_search.best_estimator_
    
    # Predictions
    y_pred_train = best_svm.predict(X_train)
    y_pred_test = best_svm.predict(X_test)
    
    # Metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"SVM Best Parameters: {grid_search.best_params_}")
    print(f"SVM Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"SVM Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    print(f"SVM Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    return best_svm, {
        'train_mse': train_mse, 'test_mse': test_mse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2,
        'best_params': grid_search.best_params_
    }

def create_ensemble_model(mlp_model, svm_model, X_test, y_test, mlp_weight=0.6, svm_weight=0.4):
    """Create ensemble model combining MLP and SVM"""
    print("Creating Ensemble Model...")
    
    # Get predictions from both models
    mlp_pred = mlp_model.predict(X_test)
    svm_pred = svm_model.predict(X_test)
    
    # Weighted ensemble
    ensemble_pred = mlp_weight * mlp_pred + svm_weight * svm_pred
    
    # Calculate ensemble metrics
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    
    print(f"Ensemble MSE: {ensemble_mse:.4f}")
    print(f"Ensemble MAE: {ensemble_mae:.4f}")
    print(f"Ensemble R²: {ensemble_r2:.4f}")
    
    return {
        'mse': ensemble_mse,
        'mae': ensemble_mae,
        'r2': ensemble_r2,
        'weights': {'mlp': mlp_weight, 'svm': svm_weight}
    }

def visualize_results(y_test, mlp_pred, svm_pred, ensemble_pred):
    """Create visualization of model results"""
    plt.figure(figsize=(15, 5))
    
    # MLP Results
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, mlp_pred, alpha=0.6, color='blue')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Quality Rating')
    plt.ylabel('Predicted Quality Rating')
    plt.title('MLP Neural Network')
    plt.grid(True, alpha=0.3)
    
    # SVM Results
    plt.subplot(1, 3, 2)
    plt.scatter(y_test, svm_pred, alpha=0.6, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Quality Rating')
    plt.ylabel('Predicted Quality Rating')
    plt.title('SVM-RBF Kernel')
    plt.grid(True, alpha=0.3)
    
    # Ensemble Results
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, ensemble_pred, alpha=0.6, color='purple')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Quality Rating')
    plt.ylabel('Predicted Quality Rating')
    plt.title('Ensemble Model')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training pipeline"""
    print("Starting Manufacturing Quality Prediction Model Training...")
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    X, y, le_material, le_operator, le_maintenance = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    mlp_model, mlp_metrics = train_mlp_model(X_train_scaled, y_train, X_test_scaled, y_test)
    svm_model, svm_metrics = train_svm_model(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Create ensemble
    mlp_pred = mlp_model.predict(X_test_scaled)
    svm_pred = svm_model.predict(X_test_scaled)
    ensemble_pred = 0.6 * mlp_pred + 0.4 * svm_pred
    
    ensemble_metrics = create_ensemble_model(mlp_model, svm_model, X_test_scaled, y_test)
    
    # Visualize results
    visualize_results(y_test, mlp_pred, svm_pred, ensemble_pred)
    
    # Save models
    joblib.dump(mlp_model, 'mlp_model.pkl')
    joblib.dump(svm_model, 'svm_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump((le_material, le_operator, le_maintenance), 'encoders.pkl')
    
    # Save metrics
    all_metrics = {
        'mlp': mlp_metrics,
        'svm': svm_metrics,
        'ensemble': ensemble_metrics
    }
    
    import json
    with open('model_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print("Training completed successfully!")
    print("Models saved: mlp_model.pkl, svm_model.pkl")
    print("Preprocessing saved: scaler.pkl, encoders.pkl")
    print("Metrics saved: model_metrics.json")
    
    return all_metrics

if __name__ == "__main__":
    metrics = main()
