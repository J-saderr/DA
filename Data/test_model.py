"""
Test script để kiểm tra XGBoost model load và predict có hoạt động đúng không
"""
import sys
from pathlib import Path

# Đảm bảo import được load_model
sys.path.insert(0, str(Path(__file__).parent))

from load_model import load_model, predict_diabetes

def test_model_loading():
    """Test xem model có load được không"""
    print("=" * 60)
    print("TEST 1: Model Loading")
    print("=" * 60)
    try:
        components = load_model()
        print("Model loaded successfully!")
        print(f"   Model type: {type(components['model']).__name__}")
        print(f"   Model name from metadata: {components['metadata'].get('best_model_name', 'Unknown')}")
        print(f"   Number of features: {len(components['feature_selector']['selected_features'])}")
        return components
    except Exception as e:
        print(f" Error loading model: {e}")
        return None


def test_prediction(components):
    """Test prediction với sample data"""
    print("\n" + "=" * 60)
    print("TEST 2: Prediction với sample data")
    print("=" * 60)
    
    if components is None:
        print(" Cannot test prediction - model not loaded")
        return
    
    # Sample input data
    test_cases = [
        {
            'name': 'High Risk Case',
            'data': {
                'HighBP': 1,
                'HighChol': 1,
                'BMI': 35,  # Obese
                'Smoker': 1,
                'PhysActivity': 0,
                'Fruits': 0,
                'GenHlth': 4,  # Poor health
                'MentHlth': 10,
                'PhysHlth': 10,
                'Age': 6,  # Older
                'Education': 2,
                'Income': 2,
                'Sex': 1
            }
        },
        {
            'name': 'Low Risk Case',
            'data': {
                'HighBP': 0,
                'HighChol': 0,
                'BMI': 22,  # Normal
                'Smoker': 0,
                'PhysActivity': 1,
                'Fruits': 1,
                'GenHlth': 2,  # Good health
                'MentHlth': 0,
                'PhysHlth': 0,
                'Age': 3,  # Middle age
                'Education': 5,
                'Income': 7,
                'Sex': 1
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        try:
            result = predict_diabetes(test_case['data'], components)
            print(f" Prediction successful!")
            print(f"   Prediction: {'Có nguy cơ' if result['prediction'] == 1 else 'Không có nguy cơ'}")
            print(f"   Probability: {result['probability']:.4f}")
            print(f"   Confidence: {result['confidence']:.4f}")
        except Exception as e:
            print(f" Prediction failed: {e}")


def test_feature_importance(components):
    """Test feature importance"""
    print("\n" + "=" * 60)
    print("TEST 3: Feature Importance")
    print("=" * 60)
    
    if components is None:
        print(" Cannot test feature importance - model not loaded")
        return
    
    model = components['model']
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        selected_features = components['feature_selector']['selected_features']
        
        # Sort by importance
        feature_importance_dict = dict(zip(selected_features, importances))
        sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
        
        print(f" Feature importance available ({len(sorted_features)} features)")
        print("\nTop 10 Most Important Features:")
        for i, (feature, importance) in enumerate(sorted_features[:10], 1):
            print(f"   {i:2d}. {feature:30s}: {importance:.6f}")
    else:
        print(" Model does not have feature_importances_ attribute")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("XGBOOST MODEL TESTING")
    print("=" * 60)
    
    # Test 1: Load model
    components = test_model_loading()
    
    # Test 2: Prediction
    test_prediction(components)
    
    # Test 3: Feature importance
    test_feature_importance(components)
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)

