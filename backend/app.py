"""
Flask Backend API cho Diabetes Prediction Web Application
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path
import random
import numpy as np

# Thêm thư mục Data vào path để import load_model
sys.path.insert(0, str(Path(__file__).parent.parent / 'Data'))

# Set random seed cho reproducibility (trước khi import các thư viện khác)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

from load_model import load_model, predict_diabetes
import pandas as pd

app = Flask(__name__)
CORS(app)  # Cho phép CORS để frontend có thể gọi API

# Load model khi khởi động app
try:
    # Đổi working directory về Data để load model
    original_dir = os.getcwd()
    data_dir = Path(__file__).parent.parent / 'Data'
    if not data_dir.exists():
        data_dir = Path(__file__).parent.parent / 'Data'
    os.chdir(data_dir)
    model_components = load_model()
    os.chdir(original_dir)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Looking for models in: {Path(__file__).parent.parent / 'Data' / 'models'}")
    model_components = None


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_components is not None
    })


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """Lấy thông tin về model"""
    if model_components is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    metadata = model_components['metadata']
    feature_selector = model_components['feature_selector']
    
    return jsonify({
        'model_name': metadata.get('best_model_name', 'Unknown'),
        'accuracy': metadata.get('accuracy', 0),
        'f1_score': metadata.get('f1_score', 0),
        'optimal_threshold': metadata.get('optimal_threshold', 0.5),
        'selected_features': feature_selector.get('selected_features', []),
        'total_features': metadata.get('total_features_count', 0),
        'selected_features_count': metadata.get('selected_features_count', 0)
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict diabetes từ input data"""
    if model_components is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Feature engineering giống như trong test.py
        input_dict = data.copy()
        
        # Tính toán interaction features
        if 'BMI' in input_dict and 'Age' in input_dict:
            input_dict['BMI_Age'] = float(input_dict['BMI']) * float(input_dict['Age'])
        if 'HighBP' in input_dict and 'HighChol' in input_dict:
            input_dict['HighBP_Chol'] = float(input_dict['HighBP']) * float(input_dict['HighChol'])
        if 'MentHlth' in input_dict and 'PhysHlth' in input_dict:
            input_dict['MentPhysHlth'] = float(input_dict['MentHlth']) + float(input_dict['PhysHlth'])
        if 'BMI' in input_dict:
            input_dict['BMI_squared'] = float(input_dict['BMI']) ** 2
        if 'Age' in input_dict:
            input_dict['Age_squared'] = float(input_dict['Age']) ** 2
        
        # Predict
        result = predict_diabetes(input_dict, model_components)
        
        # Tính feature importance nếu có
        feature_importance = None
        if hasattr(model_components['model'], 'feature_importances_'):
            feature_selector = model_components['feature_selector']
            selected_features = feature_selector['selected_features']
            importances = model_components['model'].feature_importances_
            feature_importance = dict(zip(selected_features, importances.tolist()))
        
        return jsonify({
            'success': True,
            'prediction': result['prediction'],
            'probability': result['probability'],
            'confidence': result['confidence'],
            'result_text': 'Có nguy cơ mắc bệnh tiểu đường' if result['prediction'] == 1 else 'Không có nguy cơ mắc bệnh tiểu đường',
            'recommendations': get_recommendations(input_dict, result),
            'feature_importance': feature_importance
        })
    
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


def get_recommendations(input_data, prediction_result):
    """Tạo recommendations dựa trên input và kết quả"""
    recommendations = []
    
    if prediction_result['prediction'] == 1:
        recommendations.append("⚠️ Bạn có nguy cơ mắc bệnh tiểu đường. Nên tham khảo ý kiến bác sĩ.")
    
    if input_data.get('BMI', 0) > 25:
        recommendations.append("💡 BMI của bạn cao. Nên giảm cân và tập thể dục thường xuyên.")
    
    if input_data.get('HighBP', 0) == 1:
        recommendations.append("💡 Bạn có huyết áp cao. Nên kiểm soát huyết áp và theo dõi định kỳ.")
    
    if input_data.get('HighChol', 0) == 1:
        recommendations.append("💡 Bạn có cholesterol cao. Nên điều chỉnh chế độ ăn uống.")
    
    if input_data.get('PhysActivity', 0) == 0:
        recommendations.append("💡 Nên tăng cường hoạt động thể chất ít nhất 30 phút mỗi ngày.")
    
    if input_data.get('Fruits', 0) == 0:
        recommendations.append("💡 Nên ăn nhiều trái cây và rau quả trong chế độ ăn uống.")
    
    if input_data.get('Smoker', 0) == 1:
        recommendations.append("💡 Hút thuốc làm tăng nguy cơ mắc bệnh. Nên bỏ thuốc lá.")
    
    if not recommendations:
        recommendations.append("✅ Bạn đang có lối sống lành mạnh. Hãy tiếp tục duy trì!")
    
    return recommendations


@app.route('/api/analyze', methods=['GET'])
def analyze_data():
    """Phân tích dữ liệu và trả về statistics"""
    try:
        # Đọc dữ liệu gốc
        csv_path = Path(__file__).parent.parent / 'diabetes_binary_health_indicators_BRFSS2015.csv'
        if not csv_path.exists():
            return jsonify({'error': 'Data file not found'}), 404
        
        df = pd.read_csv(csv_path)
        
        # Basic statistics
        total_samples = len(df)
        diabetes_count = df['Diabetes_binary'].sum()
        diabetes_rate = (diabetes_count / total_samples) * 100
        
        # Feature statistics
        numeric_features = ['BMI', 'MentHlth', 'PhysHlth', 'Age']
        feature_stats = {}
        for feature in numeric_features:
            if feature in df.columns:
                feature_stats[feature] = {
                    'mean': float(df[feature].mean()),
                    'median': float(df[feature].median()),
                    'std': float(df[feature].std()),
                    'min': float(df[feature].min()),
                    'max': float(df[feature].max())
                }
        
        # Categorical features distribution
        categorical_features = ['HighBP', 'HighChol', 'Smoker', 'PhysActivity', 'Fruits', 'Sex']
        categorical_dist = {}
        for feature in categorical_features:
            if feature in df.columns:
                dist = df[feature].value_counts().to_dict()
                categorical_dist[feature] = {str(k): int(v) for k, v in dist.items()}
        
        # Correlation với target
        correlations = {}
        if 'Diabetes_binary' in df.columns:
            corr_with_target = df.corr()['Diabetes_binary'].abs().sort_values(ascending=False)
            correlations = {k: float(v) for k, v in corr_with_target.head(10).items() if k != 'Diabetes_binary'}
        
        return jsonify({
            'total_samples': int(total_samples),
            'diabetes_count': int(diabetes_count),
            'diabetes_rate': float(diabetes_rate),
            'feature_stats': feature_stats,
            'categorical_distribution': categorical_dist,
            'correlations': correlations
        })
    
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500


if __name__ == '__main__':
    import os
    # Cho phép thay đổi port qua environment variable, mặc định 5001 (tránh conflict với AirPlay)
    port = int(os.environ.get('PORT', 5001))
    app.run(debug=True, host='0.0.0.0', port=port)

