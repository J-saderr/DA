# Diabetes Prediction Web Application

Ứng dụng web dự đoán bệnh tiểu đường sử dụng Machine Learning (XGBoost/LightGBM).

## Cấu Trúc Dự Án

```
DA/
├── backend/
│   ├── app.py              # Flask API server
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── index.html          # Trang chính
│   ├── css/
│   │   └── style.css       # Styles
│   └── js/
│       └── main.js         # JavaScript logic
└── Data/
    ├── models/             # Trained models (sau khi chạy test.py)
    ├── load_model.py       # Model loading helper
    ├── test.py             # Training script
    └── utils.py            # Utility functions (random seed management)
```
## Cài Đặt

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
```

### 2. Train Model (nếu chưa có)

```bash
cd Data
python test.py
```

Model sẽ được lưu vào `Data/models/`:
- `best_model.pkl` - XGBoost model đã được train
- `scaler.pkl` - RobustScaler đã được fit
- `feature_selector.pkl` - Selected features và feature importance
- `metadata.pkl` - Model metadata (accuracy, f1_score, optimal_threshold, etc.)

### 2.1. Test Model (Tùy chọn)

Để kiểm tra model có load và predict đúng không:

```bash
cd Data
python test_model.py
```

Script này sẽ:
- Load XGBoost model từ `models/`
- Test prediction với sample data
- Hiển thị feature importance

### 3. Chạy Backend Server

```bash
cd backend
python app.py
```

Server sẽ chạy tại `http://localhost:5001` (mặc định)

### 4. Mở Frontend

Mở file `frontend/index.html` trong trình duyệt hoặc sử dụng local server:

```bash
cd frontend
python -m http.server 8000
```

Truy cập: `http://localhost:8000`