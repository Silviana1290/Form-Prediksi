import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_manufacturing_data():
    """Load and preprocess manufacturing dataset"""
    # Load the CSV data
    df = pd.read_csv('data/manufacturing.csv')
    
    print(f"Dataset loaded: {len(df)} samples, {len(df.columns)} features")
    print(f"Target variable (Quality_Score) range: {df['Quality_Score'].min():.1f} - {df['Quality_Score'].max():.1f}")
    
    return df

def preprocess_data(df):
    """Preprocess the manufacturing data for ML models"""
    # Separate features and target
    X = df.drop(['ID', 'Quality_Score'], axis=1)
    y = df['Quality_Score']
    
    # Encode categorical variables
    categorical_columns = ['Material_Type', 'Maintenance_Status', 'Vibration_Level', 
                          'Dust_Level', 'Operator_Experience', 'Shift_Type', 
                          'Inspection_Level', 'Tolerance_Level']
    
    label_encoders = {}
    for col in categorical_columns:
        if col in X.columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, label_encoders, X.columns.tolist()

def train_mlp_model(X_train, y_train, X_test, y_test):
    """Train and evaluate MLP Neural Network"""
    print("Training MLP Neural Network...")
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'hidden_layer_sizes': [(100,), (100, 50), (150, 100, 50), (200, 100, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
        'max_iter': [1000, 2000]
    }
    
    mlp = MLPRegressor(random_state=42, early_stopping=True, validation_fraction=0.1)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        mlp, param_grid, cv=5, scoring='neg_mean_squared_error', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_mlp = grid_search.best_estimator_
    
    # Make predictions
    y_pred_train = best_mlp.predict(X_train)
    y_pred_test = best_mlp.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Calculate accuracy (within ±5 points)
    train_accuracy = np.mean(np.abs(y_train - y_pred_train) <= 5) * 100
    test_accuracy = np.mean(np.abs(y_test - y_pred_test) <= 5) * 100
    
    print(f"MLP Best Parameters: {grid_search.best_params_}")
    print(f"MLP Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"MLP Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    print(f"MLP Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"MLP Train Accuracy: {train_accuracy:.1f}%, Test Accuracy: {test_accuracy:.1f}%")
    
    return best_mlp, {
        'train_mse': train_mse, 'test_mse': test_mse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy,
        'best_params': grid_search.best_params_,
        'predictions': y_pred_test
    }

def train_svm_model(X_train, y_train, X_test, y_test):
    """Train and evaluate SVM with RBF kernel"""
    print("Training SVM with RBF Kernel...")
    
    # Define parameter grid for hyperparameter tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
        'epsilon': [0.01, 0.1, 0.2, 0.5]
    }
    
    svm = SVR(kernel='rbf')
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        svm, param_grid, cv=5, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_svm = grid_search.best_estimator_
    
    # Make predictions
    y_pred_train = best_svm.predict(X_train)
    y_pred_test = best_svm.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Calculate accuracy (within ±5 points)
    train_accuracy = np.mean(np.abs(y_train - y_pred_train) <= 5) * 100
    test_accuracy = np.mean(np.abs(y_test - y_pred_test) <= 5) * 100
    
    print(f"SVM Best Parameters: {grid_search.best_params_}")
    print(f"SVM Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"SVM Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    print(f"SVM Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"SVM Train Accuracy: {train_accuracy:.1f}%, Test Accuracy: {test_accuracy:.1f}%")
    
    return best_svm, {
        'train_mse': train_mse, 'test_mse': test_mse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy,
        'best_params': grid_search.best_params_,
        'predictions': y_pred_test
    }

def create_ensemble_model(mlp_model, svm_model, X_train, y_train, X_test, y_test):
    """Create and evaluate ensemble model"""
    print("Creating Ensemble Model...")
    
    # Create voting regressor with optimized weights
    ensemble = VotingRegressor([
        ('mlp', mlp_model),
        ('svm', svm_model)
    ], weights=[0.6, 0.4])  # MLP gets 60% weight, SVM gets 40%
    
    ensemble.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = ensemble.predict(X_train)
    y_pred_test = ensemble.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    # Calculate accuracy (within ±5 points)
    train_accuracy = np.mean(np.abs(y_train - y_pred_train) <= 5) * 100
    test_accuracy = np.mean(np.abs(y_test - y_pred_test) <= 5) * 100
    
    print(f"Ensemble Train MSE: {train_mse:.4f}, Test MSE: {test_mse:.4f}")
    print(f"Ensemble Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    print(f"Ensemble Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    print(f"Ensemble Train Accuracy: {train_accuracy:.1f}%, Test Accuracy: {test_accuracy:.1f}%")
    
    return ensemble, {
        'train_mse': train_mse, 'test_mse': test_mse,
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_accuracy': train_accuracy, 'test_accuracy': test_accuracy,
        'predictions': y_pred_test
    }

def evaluate_models_cross_validation(models, X, y):
    """Perform cross-validation evaluation"""
    print("Performing Cross-Validation...")
    
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        cv_results[name] = {
            'mean_mse': -scores.mean(),
            'std_mse': scores.std(),
            'scores': -scores
        }
        print(f"{name} CV MSE: {-scores.mean():.4f} (±{scores.std():.4f})")
    
    return cv_results

def plot_model_comparison(mlp_metrics, svm_metrics, ensemble_metrics):
    """Create visualization comparing model performance"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    models = ['MLP', 'SVM', 'Ensemble']
    
    # Test Accuracy
    accuracies = [mlp_metrics['test_accuracy'], svm_metrics['test_accuracy'], ensemble_metrics['test_accuracy']]
    axes[0, 0].bar(models, accuracies, color=['#3498db', '#27ae60', '#9b59b6'])
    axes[0, 0].set_title('Test Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].set_ylim(0, 100)
    
    # Test MAE
    maes = [mlp_metrics['test_mae'], svm_metrics['test_mae'], ensemble_metrics['test_mae']]
    axes[0, 1].bar(models, maes, color=['#3498db', '#27ae60', '#9b59b6'])
    axes[0, 1].set_title('Test MAE Comparison')
    axes[0, 1].set_ylabel('Mean Absolute Error')
    
    # Test R²
    r2s = [mlp_metrics['test_r2'], svm_metrics['test_r2'], ensemble_metrics['test_r2']]
    axes[1, 0].bar(models, r2s, color=['#3498db', '#27ae60', '#9b59b6'])
    axes[1, 0].set_title('Test R² Score Comparison')
    axes[1, 0].set_ylabel('R² Score')
    axes[1, 0].set_ylim(0, 1)
    
    # Test MSE
    mses = [mlp_metrics['test_mse'], svm_metrics['test_mse'], ensemble_metrics['test_mse']]
    axes[1, 1].bar(models, mses, color=['#3498db', '#27ae60', '#9b59b6'])
    axes[1, 1].set_title('Test MSE Comparison')
    axes[1, 1].set_ylabel('Mean Squared Error')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_final_report(mlp_metrics, svm_metrics, ensemble_metrics, feature_names):
    """Generate comprehensive evaluation report"""
    report = f"""
# LAPORAN EVALUASI MODEL MACHINE LEARNING
## Prediksi Performa Industri Manufaktur

### RINGKASAN EKSEKUTIF
Penelitian ini mengimplementasikan dan membandingkan dua pendekatan machine learning:
1. Multi-Layer Perceptron (MLP) Neural Network
2. Support Vector Machine (SVM) dengan RBF Kernel
3. Ensemble Model (kombinasi MLP dan SVM)

### DATASET
- Total sampel: 50 data manufaktur
- Jumlah fitur: {len(feature_names)} parameter
- Target: Quality Score (0-100)

### HASIL EVALUASI MODEL

#### 1. MLP Neural Network
- Test Accuracy: {mlp_metrics['test_accuracy']:.1f}%
- Test MAE: {mlp_metrics['test_mae']:.2f}
- Test R² Score: {mlp_metrics['test_r2']:.3f}
- Test MSE: {mlp_metrics['test_mse']:.2f}

#### 2. SVM-RBF Kernel
- Test Accuracy: {svm_metrics['test_accuracy']:.1f}%
- Test MAE: {svm_metrics['test_mae']:.2f}
- Test R² Score: {svm_metrics['test_r2']:.3f}
- Test MSE: {svm_metrics['test_mse']:.2f}

#### 3. Ensemble Model
- Test Accuracy: {ensemble_metrics['test_accuracy']:.1f}%
- Test MAE: {ensemble_metrics['test_mae']:.2f}
- Test R² Score: {ensemble_metrics['test_r2']:.3f}
- Test MSE: {ensemble_metrics['test_mse']:.2f}

### KESIMPULAN
Model terbaik: {"Ensemble" if ensemble_metrics['test_accuracy'] >= max(mlp_metrics['test_accuracy'], svm_metrics['test_accuracy']) else "MLP" if mlp_metrics['test_accuracy'] > svm_metrics['test_accuracy'] else "SVM"}

### REKOMENDASI
1. Implementasi model ensemble untuk prediksi produksi
2. Monitoring real-time parameter kritis
3. Optimasi parameter proses berdasarkan prediksi model
4. Pelatihan operator untuk konsistensi kualitas

Tanggal Evaluasi: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    """
    
    with open('laporan_evaluasi_model.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("Laporan evaluasi telah disimpan ke 'laporan_evaluasi_model.txt'")
    return report

def main():
    """Main evaluation pipeline"""
    print("=== EVALUASI MODEL MACHINE LEARNING ===")
    print("Prediksi Performa Industri Manufaktur")
    print("Kelompok 3 - Kelas D")
    print("=" * 50)
    
    # Load and preprocess data
    df = load_manufacturing_data()
    X, y, scaler, label_encoders, feature_names = preprocess_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=pd.cut(y, bins=3)
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train models
    mlp_model, mlp_metrics = train_mlp_model(X_train, y_train, X_test, y_test)
    svm_model, svm_metrics = train_svm_model(X_train, y_train, X_test, y_test)
    ensemble_model, ensemble_metrics = create_ensemble_model(
        mlp_model, svm_model, X_train, y_train, X_test, y_test
    )
    
    # Cross-validation
    models = {
        'MLP': mlp_model,
        'SVM': svm_model,
        'Ensemble': ensemble_model
    }
    cv_results = evaluate_models_cross_validation(models, X, y)
    
    # Create visualizations
    plot_model_comparison(mlp_metrics, svm_metrics, ensemble_metrics)
    
    # Save models
    joblib.dump(mlp_model, 'models/mlp_model.pkl')
    joblib.dump(svm_model, 'models/svm_model.pkl')
    joblib.dump(ensemble_model, 'models/ensemble_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    
    # Generate final report
    report = generate_final_report(mlp_metrics, svm_metrics, ensemble_metrics, feature_names)
    
    # Save all metrics
    all_metrics = {
        'mlp': mlp_metrics,
        'svm': svm_metrics,
        'ensemble': ensemble_metrics,
        'cross_validation': cv_results,
        'feature_names': feature_names
    }
    
    import json
    with open('model_evaluation_results.json', 'w') as f:
        json.dump(all_metrics, f, indent=2, default=str)
    
    print("\n=== EVALUASI SELESAI ===")
    print("File yang dihasilkan:")
    print("- model_comparison.png")
    print("- laporan_evaluasi_model.txt")
    print("- model_evaluation_results.json")
    print("- models/mlp_model.pkl")
    print("- models/svm_model.pkl")
    print("- models/ensemble_model.pkl")
    
    return all_metrics

if __name__ == "__main__":
    metrics = main()
