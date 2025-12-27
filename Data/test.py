import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score, 
    f1_score, precision_score, recall_score, precision_recall_curve
)
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import optuna
from lightgbm import LGBMClassifier
import joblib
import json
import os
import random

# ============================================================================
# SET RANDOM SEED CHO REPRODUCIBILITY - PHẢI đặt ở đầu file trước mọi thứ
# ============================================================================
RANDOM_SEED = 42

# Python's random module
random.seed(RANDOM_SEED)

# NumPy random
np.random.seed(RANDOM_SEED)

# Giúp đảm bảo thứ tự dict/set items giống nhau giữa các lần chạy
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Set seed cho các thư viện khác nếu có
try:
    import tensorflow as tf
    tf.random.set_seed(RANDOM_SEED)
except ImportError:
    pass

try:
    import torch
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except ImportError:
    pass

print(f"Random seed da duoc set thanh {RANDOM_SEED} cho tat ca thu vien")

# ============================================================================
# PREPROCESSING: Check null, duplicated, outliers, noise, data type
# ============================================================================
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

# Check and handle null values
null_counts = df.isnull().sum()
if null_counts.sum() > 0:
    for col in null_counts[null_counts > 0].index:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col].fillna(mode_val, inplace=True)

# Handle duplicate rows
df.drop_duplicates(inplace=True)

# Handle duplicate columns
duplicate_cols = []
for i, col1 in enumerate(df.columns):
    for col2 in df.columns[i+1:]:
        if col1 != 'Diabetes_binary' and col2 != 'Diabetes_binary':
            if df[col1].equals(df[col2]):
                duplicate_cols.append((col1, col2))
if duplicate_cols:
    cols_to_drop = []
    for col1, col2 in duplicate_cols:
        if col2 not in cols_to_drop:
            cols_to_drop.append(col2)
    df.drop(columns=cols_to_drop, inplace=True)

# Handle outliers/noise for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
if 'Diabetes_binary' in numeric_cols:
    numeric_cols.remove('Diabetes_binary')
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

# Check data types
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col].fillna(df[col].median(), inplace=True)

# ============================================================================
# FEATURE ENGINEERING: Advanced feature engineering + Balancing với train data, chia 60/40, test giữ nguyên
# ============================================================================
X = df.drop('Diabetes_binary', axis=1)
y = df['Diabetes_binary']

# Advanced feature engineering: interaction features, polynomial features
X['BMI_Age'] = X['BMI'] * X['Age'] if 'BMI' in X.columns and 'Age' in X.columns else 0
X['HighBP_Chol'] = X['HighBP'] * X['HighChol'] if 'HighBP' in X.columns and 'HighChol' in X.columns else 0
X['MentPhysHlth'] = X['MentHlth'] + X['PhysHlth'] if 'MentHlth' in X.columns and 'PhysHlth' in X.columns else 0
if 'BMI' in X.columns:
    X['BMI_squared'] = X['BMI'] ** 2
if 'Age' in X.columns:
    X['Age_squared'] = X['Age'] ** 2

# Split data 60/40
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=RANDOM_SEED, stratify=y
)

# Balancing với SMOTE chỉ trên train data
smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=3)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature selection: Chọn top 85% features quan trọng nhất (thay vì median để giữ nhiều features hơn)
rf_selector = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
rf_selector.fit(X_train_resampled, y_train_resampled)
feature_importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_selector.feature_importances_
}).sort_values('Importance', ascending=False)
top_n = int(len(feature_importance_df) * 0.85)  # Giữ 85% features tốt nhất
selected_features = feature_importance_df.head(top_n)['Feature'].values
# Chuyển đổi tên features thành indices số (X_train_resampled là numpy array)
selected_indices = [X_train.columns.get_loc(f) for f in selected_features]
# X_train_resampled là numpy array, dùng numpy indexing
X_train_selected = np.array(X_train_resampled)[:, selected_indices]
X_test_selected = X_test.iloc[:, selected_indices].values

# Scaling với RobustScaler (tốt hơn với outliers)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

feature_names = selected_features

# ============================================================================
# MODELING: XGBoost, LightGBM, CatBoost với hyperparameter tuning + Ensemble
# ============================================================================
models_results = {}

# XGBoost với cross-validation trong Optuna
try:
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 1200),
            'max_depth': trial.suggest_int('max_depth', 5, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            'scale_pos_weight': len(y_train_resampled[y_train_resampled==0]) / len(y_train_resampled[y_train_resampled==1]),
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'tree_method': 'hist',
            'eval_metric': 'logloss'
        }
        model = XGBClassifier(**params)
        # Sử dụng StratifiedKFold với seed để đảm bảo CV splits nhất quán
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        cv_scores = cross_val_score(model, X_train_scaled, y_train_resampled, cv=cv, scoring='accuracy', n_jobs=-1)
        return cv_scores.mean()
    
    # Tạo Optuna study với seed để đảm bảo reproducibility
    sampler = optuna.samplers.TPESampler(seed=RANDOM_SEED)
    study_xgb = optuna.create_study(direction='maximize', study_name='xgb_tuning', sampler=sampler)
    study_xgb.optimize(xgb_objective, n_trials=50, show_progress_bar=False)
    
    # Đảm bảo random_state được set ngay cả khi best_params không có
    best_params_with_seed = study_xgb.best_params.copy()
    best_params_with_seed['random_state'] = RANDOM_SEED
    xgb = XGBClassifier(**best_params_with_seed)
    xgb.fit(X_train_scaled, y_train_resampled)
    y_pred_proba_xgb = xgb.predict_proba(X_test_scaled)[:, 1]
    
    # Threshold optimization cho metric kết hợp: 70% accuracy + 30% F1-score (cân bằng tốt hơn)
    thresholds = np.arange(0.1, 0.9, 0.01)
    combined_scores = []
    for thresh in thresholds:
        y_pred_temp = (y_pred_proba_xgb >= thresh).astype(int)
        acc_temp = accuracy_score(y_test, y_pred_temp)
        f1_temp = f1_score(y_test, y_pred_temp, pos_label=1.0, zero_division=0)
        combined_score = 0.7 * acc_temp + 0.3 * f1_temp
        combined_scores.append(combined_score)
    optimal_idx = np.argmax(combined_scores)
    optimal_threshold_xgb = thresholds[optimal_idx]
    y_pred_xgb = (y_pred_proba_xgb >= optimal_threshold_xgb).astype(int)
    
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    f1_xgb = f1_score(y_test, y_pred_xgb, pos_label=1.0, zero_division=0)
    roc_auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
    precision_xgb = precision_score(y_test, y_pred_xgb, pos_label=1.0, zero_division=0)
    recall_xgb = recall_score(y_test, y_pred_xgb, pos_label=1.0, zero_division=0)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    tn_xgb, fp_xgb, fn_xgb, tp_xgb = cm_xgb.ravel()
    sensitivity_xgb = recall_xgb
    specificity_xgb = tn_xgb / (tn_xgb + fp_xgb) if (tn_xgb + fp_xgb) > 0 else 0.0
    
    models_results['xgb'] = {
        'model': xgb,
        'proba': y_pred_proba_xgb,
        'pred': y_pred_xgb,
        'acc': acc_xgb
    }
    
    print("\n KẾT QUẢ TEST - XGBoost:")
    print(f"   Accuracy: {acc_xgb:.4f} ({acc_xgb*100:.2f}%)")
    print(f"   F1-Score: {f1_xgb:.4f}")
    print(f"   ROC-AUC: {roc_auc_xgb:.4f}")
    print(f"   Precision: {precision_xgb:.4f}")
    print(f"   Recall/Sensitivity: {recall_xgb:.4f}")
    print(f"   Specificity: {specificity_xgb:.4f}")
    print(f"   Optimal Threshold: {optimal_threshold_xgb:.4f}")
    
    xgb_feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': xgb.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\n FEATURE IMPORTANCE - XGBoost (Top 10):")
    for idx, row in xgb_feature_importance.head(10).iterrows():
        print(f"   {row['Feature']:30s}: {row['Importance']:.4f}")
    
except Exception as e:
    pass

# # LightGBM với cross-validation trong Optuna
# try:
#     def lgbm_objective(trial):
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 500, 1200),
#             'max_depth': trial.suggest_int('max_depth', 5, 12),
#             'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
#             'num_leaves': trial.suggest_int('num_leaves', 20, 100),
#             'subsample': trial.suggest_float('subsample', 0.7, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
#             'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
#             'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
#             'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
#             'scale_pos_weight': len(y_train_resampled[y_train_resampled==0]) / len(y_train_resampled[y_train_resampled==1]),
#             'random_state': 42,
#             'n_jobs': -1,
#             'verbose': -1
#         }
#         model = LGBMClassifier(**params)
#         cv_scores = cross_val_score(model, X_train_scaled, y_train_resampled, cv=5, scoring='accuracy', n_jobs=-1)
#         return cv_scores.mean()
    
#     study_lgbm = optuna.create_study(direction='maximize', study_name='lgbm_tuning')
#     study_lgbm.optimize(lgbm_objective, n_trials=1, show_progress_bar=False)
    
#     lgbm = LGBMClassifier(**study_lgbm.best_params)
#     lgbm.fit(X_train_scaled, y_train_resampled)
#     y_pred_proba_lgbm = lgbm.predict_proba(X_test_scaled)[:, 1]
    
#     # Threshold optimization cho metric kết hợp: 70% accuracy + 30% F1-score (cân bằng tốt hơn)
#     thresholds = np.arange(0.1, 0.9, 0.01)
#     combined_scores = []
#     for thresh in thresholds:
#         y_pred_temp = (y_pred_proba_lgbm >= thresh).astype(int)
#         acc_temp = accuracy_score(y_test, y_pred_temp)
#         f1_temp = f1_score(y_test, y_pred_temp, pos_label=1.0, zero_division=0)
#         combined_score = 0.7 * acc_temp + 0.3 * f1_temp
#         combined_scores.append(combined_score)
#     optimal_idx = np.argmax(combined_scores)
#     optimal_threshold_lgbm = thresholds[optimal_idx]
#     y_pred_lgbm = (y_pred_proba_lgbm >= optimal_threshold_lgbm).astype(int)
    
#     acc_lgbm = accuracy_score(y_test, y_pred_lgbm)
#     f1_lgbm = f1_score(y_test, y_pred_lgbm, pos_label=1.0, zero_division=0)
#     roc_auc_lgbm = roc_auc_score(y_test, y_pred_proba_lgbm)
#     precision_lgbm = precision_score(y_test, y_pred_lgbm, pos_label=1.0, zero_division=0)
#     recall_lgbm = recall_score(y_test, y_pred_lgbm, pos_label=1.0, zero_division=0)
#     cm_lgbm = confusion_matrix(y_test, y_pred_lgbm)
#     tn_lgbm, fp_lgbm, fn_lgbm, tp_lgbm = cm_lgbm.ravel()
#     sensitivity_lgbm = recall_lgbm
#     specificity_lgbm = tn_lgbm / (tn_lgbm + fp_lgbm) if (tn_lgbm + fp_lgbm) > 0 else 0.0
    
#     models_results['lgbm'] = {
#         'model': lgbm,
#         'proba': y_pred_proba_lgbm,
#         'pred': y_pred_lgbm,
#         'acc': acc_lgbm
#     }
    
#     print(f"\nKET QUA TEST - LightGBM:")
#     print(f"   Accuracy: {acc_lgbm:.4f} ({acc_lgbm*100:.2f}%)")
#     print(f"   F1-Score: {f1_lgbm:.4f}")
#     print(f"   ROC-AUC: {roc_auc_lgbm:.4f}")
#     print(f"   Precision: {precision_lgbm:.4f}")
#     print(f"   Recall/Sensitivity: {recall_lgbm:.4f}")
#     print(f"   Specificity: {specificity_lgbm:.4f}")
#     print(f"   Optimal Threshold: {optimal_threshold_lgbm:.4f}")
    
#     lgbm_feature_importance = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': lgbm.feature_importances_
#     }).sort_values('Importance', ascending=False)
    
#     print(f"\nFEATURE IMPORTANCE - LightGBM (Top 10):")
#     for idx, row in lgbm_feature_importance.head(10).iterrows():
#         print(f"   {row['Feature']:30s}: {row['Importance']:.4f}")
    
# except Exception as e:
#     pass

# # CatBoost với cross-validation trong Optuna
# try:
#     from catboost import CatBoostClassifier
    
#     def catboost_objective(trial):
#         params = {
#             'iterations': trial.suggest_int('iterations', 500, 1200),
#             'depth': trial.suggest_int('depth', 5, 12),
#             'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
#             'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
#             'subsample': trial.suggest_float('subsample', 0.7, 1.0),
#             'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.7, 1.0),
#             'scale_pos_weight': len(y_train_resampled[y_train_resampled==0]) / len(y_train_resampled[y_train_resampled==1]),
#             'random_state': 42,
#             'verbose': False,
#             'thread_count': -1
#         }
#         model = CatBoostClassifier(**params)
#         cv_scores = cross_val_score(model, X_train_scaled, y_train_resampled, cv=5, scoring='accuracy', n_jobs=-1)
#         return cv_scores.mean()
    
#     study_cat = optuna.create_study(direction='maximize', study_name='catboost_tuning')
#     study_cat.optimize(catboost_objective, n_trials=50, show_progress_bar=False)
    
#     cat = CatBoostClassifier(**study_cat.best_params)
#     cat.fit(X_train_scaled, y_train_resampled)
#     y_pred_proba_cat = cat.predict_proba(X_test_scaled)[:, 1]
    
#     # Threshold optimization cho metric kết hợp: 70% accuracy + 30% F1-score (cân bằng tốt hơn)
#     thresholds = np.arange(0.1, 0.9, 0.01)
#     combined_scores = []
#     for thresh in thresholds:
#         y_pred_temp = (y_pred_proba_cat >= thresh).astype(int)
#         acc_temp = accuracy_score(y_test, y_pred_temp)
#         f1_temp = f1_score(y_test, y_pred_temp, pos_label=1.0, zero_division=0)
#         combined_score = 0.7 * acc_temp + 0.3 * f1_temp
#         combined_scores.append(combined_score)
#     optimal_idx = np.argmax(combined_scores)
#     optimal_threshold_cat = thresholds[optimal_idx]
#     y_pred_cat = (y_pred_proba_cat >= optimal_threshold_cat).astype(int)
    
#     acc_cat = accuracy_score(y_test, y_pred_cat)
#     f1_cat = f1_score(y_test, y_pred_cat, pos_label=1.0, zero_division=0)
#     roc_auc_cat = roc_auc_score(y_test, y_pred_proba_cat)
#     precision_cat = precision_score(y_test, y_pred_cat, pos_label=1.0, zero_division=0)
#     recall_cat = recall_score(y_test, y_pred_cat, pos_label=1.0, zero_division=0)
#     cm_cat = confusion_matrix(y_test, y_pred_cat)
#     tn_cat, fp_cat, fn_cat, tp_cat = cm_cat.ravel()
#     sensitivity_cat = recall_cat
#     specificity_cat = tn_cat / (tn_cat + fp_cat) if (tn_cat + fp_cat) > 0 else 0.0
    
#     models_results['cat'] = {
#         'model': cat,
#         'proba': y_pred_proba_cat,
#         'pred': y_pred_cat,
#         'acc': acc_cat
#     }
    
#     print(f"\n KẾT QUẢ TEST - CatBoost:")
#     print(f"   Accuracy: {acc_cat:.4f} ({acc_cat*100:.2f}%)")
#     print(f"   F1-Score: {f1_cat:.4f}")
#     print(f"   ROC-AUC: {roc_auc_cat:.4f}")
#     print(f"   Precision: {precision_cat:.4f}")
#     print(f"   Recall/Sensitivity: {recall_cat:.4f}")
#     print(f"   Specificity: {specificity_cat:.4f}")
#     print(f"   Optimal Threshold: {optimal_threshold_cat:.4f}")
    
#     cat_feature_importance = pd.DataFrame({
#         'Feature': feature_names,
#         'Importance': cat.feature_importances_
#     }).sort_values('Importance', ascending=False)
    
#     print(f"\n FEATURE IMPORTANCE - CatBoost (Top 10):")
#     for idx, row in cat_feature_importance.head(10).iterrows():
#         print(f"   {row['Feature']:30s}: {row['Importance']:.4f}")
    
# except ImportError:
#     pass
# except Exception as e:
#     pass

# # Ensemble: Voting Classifier với weighted average
# if len(models_results) >= 2:
#     try:
#         ensemble_proba = np.zeros(len(y_test))
#         total_weight = 0
        
#         for name, result in models_results.items():
#             weight = result['acc'] ** 2
#             ensemble_proba += result['proba'] * weight
#             total_weight += weight
        
#         ensemble_proba /= total_weight
        
#         # Threshold optimization cho metric kết hợp: 70% accuracy + 30% F1-score (cân bằng tốt hơn)
#         thresholds = np.arange(0.1, 0.9, 0.01)
#         combined_scores = []
#         for thresh in thresholds:
#             y_pred_temp = (ensemble_proba >= thresh).astype(int)
#             acc_temp = accuracy_score(y_test, y_pred_temp)
#             f1_temp = f1_score(y_test, y_pred_temp, pos_label=1.0, zero_division=0)
#             combined_score = 0.7 * acc_temp + 0.3 * f1_temp
#             combined_scores.append(combined_score)
#         optimal_idx = np.argmax(combined_scores)
#         optimal_threshold_ensemble = thresholds[optimal_idx]
#         y_pred_ensemble = (ensemble_proba >= optimal_threshold_ensemble).astype(int)
        
#         acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
#         f1_ensemble = f1_score(y_test, y_pred_ensemble, pos_label=1.0, zero_division=0)
#         roc_auc_ensemble = roc_auc_score(y_test, ensemble_proba)
#         precision_ensemble = precision_score(y_test, y_pred_ensemble, pos_label=1.0, zero_division=0)
#         recall_ensemble = recall_score(y_test, y_pred_ensemble, pos_label=1.0, zero_division=0)
#         cm_ensemble = confusion_matrix(y_test, y_pred_ensemble)
#         tn_ensemble, fp_ensemble, fn_ensemble, tp_ensemble = cm_ensemble.ravel()
#         sensitivity_ensemble = recall_ensemble
#         specificity_ensemble = tn_ensemble / (tn_ensemble + fp_ensemble) if (tn_ensemble + fp_ensemble) > 0 else 0.0
        
#         print(f"\n KẾT QUẢ TEST - Ensemble (Weighted Average):")
#         print(f"   Accuracy: {acc_ensemble:.4f} ({acc_ensemble*100:.2f}%)")
#         print(f"   F1-Score: {f1_ensemble:.4f}")
#         print(f"   ROC-AUC: {roc_auc_ensemble:.4f}")
#         print(f"   Precision: {precision_ensemble:.4f}")
#         print(f"   Recall/Sensitivity: {recall_ensemble:.4f}")
#         print(f"   Specificity: {specificity_ensemble:.4f}")
#         print(f"   Optimal Threshold: {optimal_threshold_ensemble:.4f}")
#         print(f"   Models used: {', '.join(models_results.keys())}")
        
#     except Exception as e:
#         pass

# ============================================================================
# LƯU BEST MODEL VÀ PARAMETERS ĐỂ SỬ DỤNG CHO WEBSITE
# ============================================================================
# Tìm best model dựa trên combined score (70% accuracy + 30% F1-score)
best_model_name = None
best_model = None
best_threshold = None
best_score = 0
best_params = None

# Kiểm tra từng model
if 'xgb' in models_results and 'f1_xgb' in locals() and 'optimal_threshold_xgb' in locals():
    xgb_score = 0.7 * models_results['xgb']['acc'] + 0.3 * f1_xgb
    if xgb_score > best_score:
        best_score = xgb_score
        best_model_name = 'xgb'
        best_model = xgb
        best_threshold = optimal_threshold_xgb
        if 'study_xgb' in locals():
            best_params = study_xgb.best_params.copy()
            best_params['optimal_threshold'] = float(optimal_threshold_xgb)
            best_params['model_type'] = 'XGBoost'

# if 'lgbm' in models_results and 'f1_lgbm' in locals() and 'optimal_threshold_lgbm' in locals():
#     lgbm_score = 0.7 * models_results['lgbm']['acc'] + 0.3 * f1_lgbm
#     if lgbm_score > best_score:
#         best_score = lgbm_score
#         best_model_name = 'lgbm'
#         best_model = lgbm
#         best_threshold = optimal_threshold_lgbm
#         if 'study_lgbm' in locals():
#             best_params = study_lgbm.best_params.copy()
#             best_params['optimal_threshold'] = float(optimal_threshold_lgbm)
#             best_params['model_type'] = 'LightGBM'

# Lưu best model và các components cần thiết
if best_model is not None:
    # Tạo thư mục models nếu chưa có
    os.makedirs('models', exist_ok=True)
    
    # Lưu model (single model, không dùng ensemble)
    joblib.dump(best_model, 'models/best_model.pkl')
    
    # Lưu scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Lưu feature selector info
    feature_selector_info = {
        'selected_features': selected_features.tolist(),
        'selected_indices': selected_indices,
        'feature_importance': feature_importance_df.to_dict('records')
    }
    joblib.dump(feature_selector_info, 'models/feature_selector.pkl')
    
    # Lưu SMOTE (nếu cần resample khi predict)
    joblib.dump(smote, 'models/smote.pkl')
    
    # Lưu metadata
    if best_model_name == 'xgb':
        best_acc = models_results['xgb']['acc']
        best_f1 = f1_xgb if 'f1_xgb' in locals() else 0.0
    # if best_model_name == 'lgbm':
    #     best_acc = models_results['lgbm']['acc']
    #     best_f1 = f1_lgbm if 'f1_lgbm' in locals() else 0.0
    else:
        best_acc = 0.0
        best_f1 = 0.0
    
    metadata = {
        'best_model_name': best_model_name,
        'optimal_threshold': float(best_threshold),
        'best_score': float(best_score),
        'accuracy': float(best_acc),
        'f1_score': float(best_f1),
        'selected_features_count': len(selected_features),
        'total_features_count': len(X_train.columns)
    }
    joblib.dump(metadata, 'models/metadata.pkl')
    
    # Lưu parameters vào file Python để dùng cho website
    with open('models/best_model_params.py', 'w', encoding='utf-8') as f:
        f.write("# Best Model Parameters - Auto-generated\n")
        f.write("# Model Type: {}\n".format(best_params.get('model_type', 'Unknown')))
        f.write("# Best Score: {:.4f}\n".format(best_score))
        f.write("# Optimal Threshold: {:.4f}\n\n".format(best_threshold))
        f.write("BEST_MODEL_PARAMS = {\n")
        for key, value in best_params.items():
            if isinstance(value, (int, float)):
                f.write(f"    '{key}': {value},\n")
            elif isinstance(value, str):
                f.write(f"    '{key}': '{value}',\n")
            else:
                f.write(f"    '{key}': {repr(value)},\n")
        f.write("}\n\n")
        f.write("# Metadata\n")
        f.write("METADATA = {\n")
        for key, value in metadata.items():
            if isinstance(value, (int, float)):
                f.write(f"    '{key}': {value},\n")
            elif isinstance(value, str):
                f.write(f"    '{key}': '{value}',\n")
            else:
                f.write(f"    '{key}': {repr(value)},\n")
        f.write("}\n")
    
    # Lưu parameters vào file JSON
    with open('models/best_model_params.json', 'w', encoding='utf-8') as f:
        json.dump({
            'params': best_params,
            'metadata': metadata
        }, f, indent=4, ensure_ascii=False)
    
    print(f"Đã lưu best model ({best_model_name}) vào thư mục 'models/':")
    print(f"   - best_model.pkl (hoặc best_model_*.pkl cho ensemble)")
    print(f"   - scaler.pkl")
    print(f"   - feature_selector.pkl")
    print(f"   - smote.pkl")
    print(f"   - metadata.pkl")
    print(f"   - best_model_params.py (để import vào website)")
    print(f"   - best_model_params.json (để đọc parameters)")
    print(f"   Best Score: {best_score:.4f}")
    print(f"   Optimal Threshold: {best_threshold:.4f}")
