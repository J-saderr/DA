"""
Helper script để load và sử dụng best model đã được train
Sử dụng trong website để predict diabetes
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import random
import os

# Fix compatibility issue between LightGBM 4.1.0 and scikit-learn 1.8.0+
# LightGBM uses force_all_finite parameter which was removed in sklearn 1.8.0
try:
    import sklearn.utils.validation as skval
    original_check_array = skval.check_array
    
    def patched_check_array(*args, **kwargs):
        # Remove force_all_finite if present (not supported in sklearn 1.8.0+)
        if 'force_all_finite' in kwargs:
            kwargs.pop('force_all_finite')
        return original_check_array(*args, **kwargs)
    
    # Apply patch globally
    skval.check_array = patched_check_array
except ImportError:
    pass

# Set random seed cho reproducibility (nếu cần predict với randomness)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Load các components
MODEL_DIR = Path('models')

def load_model():
    import warnings
    # Suppress version warnings khi load model (model vẫn hoạt động đúng)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        
        try:
            # Load metadata để biết model type
            metadata = joblib.load(MODEL_DIR / 'metadata.pkl')
            model_name = metadata.get('best_model_name', 'Unknown')
            
            # Load XGBoost model (suppress pickle warning - model vẫn hoạt động đúng)
            model = joblib.load(MODEL_DIR / 'best_model.pkl')
            
            # Load các components khác
            scaler = joblib.load(MODEL_DIR / 'scaler.pkl')
            feature_selector = joblib.load(MODEL_DIR / 'feature_selector.pkl')
            
            # Verify model type
            model_type = type(model).__name__
            print(f"Loaded {model_type} model (metadata: {model_name})")
            print(f"   Selected features: {len(feature_selector['selected_features'])}")
            print(f"   Accuracy: {metadata.get('accuracy', 0):.4f}")
            print(f"   Optimal threshold: {metadata.get('optimal_threshold', 0.5):.4f}")
            
            return {
                'model': model,
                'scaler': scaler,
                'feature_selector': feature_selector,
                'metadata': metadata
            }
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Không tìm thấy file model trong {MODEL_DIR}. Hãy chạy test.py để train model trước.")
        except Exception as e:
            raise Exception(f"Lỗi khi load model: {str(e)}")


def predict_diabetes(input_data, model_components=None):
    if model_components is None:
        model_components = load_model()
    
    model = model_components['model']
    scaler = model_components['scaler']
    feature_selector = model_components['feature_selector']
    metadata = model_components['metadata']
    
    # Chuyển input_data thành DataFrame nếu là dict
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])
    else:
        df = input_data.copy()
    
    # Đảm bảo có đủ features - fill missing với giá trị mặc định hợp lý
    selected_features = feature_selector['selected_features']
    missing_features = set(selected_features) - set(df.columns)
    
    if missing_features:
        # Fill missing features với giá trị mặc định (0 cho binary, median cho numeric)
        for feature in missing_features:
            # Nếu là interaction feature hoặc polynomial feature, tính từ features gốc
            if feature == 'BMI_Age' and 'BMI' in df.columns and 'Age' in df.columns:
                df[feature] = df['BMI'] * df['Age']
            elif feature == 'HighBP_Chol' and 'HighBP' in df.columns and 'HighChol' in df.columns:
                df[feature] = df['HighBP'] * df['HighChol']
            elif feature == 'MentPhysHlth' and 'MentHlth' in df.columns and 'PhysHlth' in df.columns:
                df[feature] = df['MentHlth'] + df['PhysHlth']
            elif feature == 'BMI_squared' and 'BMI' in df.columns:
                df[feature] = df['BMI'] ** 2
            elif feature == 'Age_squared' and 'Age' in df.columns:
                df[feature] = df['Age'] ** 2
            else:
                # Default: 0 cho các features khác
                df[feature] = 0
                print(f"Warning: Feature '{feature}' không có trong input, sử dụng giá trị mặc định 0")
    
    # Chọn đúng features và đúng thứ tự theo thứ tự trong selected_features
    # Ensure all selected_features exist in df
    for feature in selected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Select features in the exact order expected by the model
    try:
        X = df[selected_features].values
    except KeyError as e:
        missing = [f for f in selected_features if f not in df.columns]
        raise KeyError(f"Missing required features: {missing}. Available: {list(df.columns)}")
    
    # Ensure data is clean (no NaN or inf values) before scaling
    # Replace any NaN or inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Ensure X is a 2D array with correct shape
    if len(X.shape) == 1:
        X = X.reshape(1, -1)
    
    # Verify shape matches expected number of features
    if X.shape[1] != len(selected_features):
        raise ValueError(f"Feature count mismatch: expected {len(selected_features)} features, got {X.shape[1]}")
    
    # Scale data với scaler đã được train
    # Handle scikit-learn version compatibility issues
    try:
        X_scaled = scaler.transform(X)
    except (TypeError, ValueError) as e:
        error_msg = str(e).lower()
        if 'force_all_finite' in error_msg or 'unexpected keyword' in error_msg:
            # Compatibility issue with scikit-learn versions
            # Try to work around by ensuring clean data and using alternative method
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            # Ensure proper dtype
            X_clean = X_clean.astype(np.float64)
            # Try transform again with clean data
            try:
                X_scaled = scaler.transform(X_clean)
            except Exception as e2:
                # Last resort: manually apply scaling if scaler has the attributes
                # RobustScaler uses center_ (median) and scale_ (IQR)
                if hasattr(scaler, 'center_') and hasattr(scaler, 'scale_'):
                    X_scaled = (X_clean - scaler.center_) / scaler.scale_
                # StandardScaler uses mean_ and scale_
                elif hasattr(scaler, 'mean_') and hasattr(scaler, 'scale_'):
                    X_scaled = (X_clean - scaler.mean_) / scaler.scale_
                else:
                    # If we can't manually transform, raise the original error
                    raise Exception(f"Scaler transformation failed: {str(e2)}. Scaler type: {type(scaler).__name__}")
        else:
            raise
    
    # Predict với model (XGBoost or LightGBM)
    # The compatibility patch is applied at module import level
    proba = model.predict_proba(X_scaled)[:, 1]
    
    # Apply optimal threshold từ metadata
    optimal_threshold = metadata['optimal_threshold']
    prediction = (proba >= optimal_threshold).astype(int)
    
    return {
        'prediction': int(prediction[0]) if len(prediction) == 1 else prediction.tolist(),
        'probability': float(proba[0]) if len(proba) == 1 else proba.tolist(),
        'confidence': float(abs(proba[0] - 0.5) * 2) if len(proba) == 1 else (abs(proba - 0.5) * 2).tolist()
    }


# Example usage
if __name__ == "__main__":
    # Load model
    components = load_model()
    print("Model loaded successfully!")
    print(f"Model type: {components['metadata']['best_model_name']}")
    print(f"Accuracy: {components['metadata']['accuracy']:.4f}")
    print(f"Optimal threshold: {components['metadata']['optimal_threshold']:.4f}")
    
    # Example prediction
    example_input = {
        'HighBP': 1,
        'HighChol': 1,
        'BMI': 30,
        'Smoker': 0,
        'PhysActivity': 1,
        'Fruits': 1,
        'GenHlth': 3,
        'MentHlth': 5,
        'PhysHlth': 5,
        'Age': 5,
        'Education': 4,
        'Income': 6,
        'Sex': 1
    }
    
    result = predict_diabetes(example_input, components)
    print(f"\nPrediction Result:")
    print(f"   Prediction: {'Diabetes' if result['prediction'] == 1 else 'No Diabetes'}")
    print(f"   Probability: {result['probability']:.4f}")
    print(f"   Confidence: {result['confidence']:.4f}")

