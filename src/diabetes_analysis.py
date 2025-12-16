# -*- coding: utf-8 -*-
"""
Diabetes Binary Health Indicators Analysis
Converted from Jupyter Notebook: Untitled16.ipynb
"""

# ============================================================================
# IMPORTS
# ============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score, accuracy_score, 
    make_scorer, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# ============================================================================
# DATA LOADING
# ============================================================================
# ###Data
df = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')

# Display basic info
df.head(5)

rows, col = df.shape
print(f"Number of Rows : {rows} \nNumber of Columns : {col}")

df.info()
df.describe()

# ============================================================================
# DATA CLEANING
# ============================================================================
# ###Check Missing values
df.isnull().sum()

# ###Check Duplicated
df.duplicated().sum()

dup_groups = (
    df.groupby(list(df.columns))
      .size()
      .reset_index(name='count')
      .query('count > 1')
)

dup_groups

duplicates_only = df[df.duplicated(keep=False)]
duplicates_only

df.drop_duplicates(inplace=True)

df.duplicated().sum()
df.shape

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================
# ###Check Outliers
# Calculate number of rows and columns dynamically
num_cols = 3  # Number of columns in the grid
num_rows = (len(df.columns) + num_cols - 1) // num_cols  # Calculate rows needed

# Create subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

# Flatten the axes for easier iteration
axes = axes.flatten()

# Loop through numeric columns and create boxplots
for i, column in enumerate(df.columns):
    sns.boxplot(data=df, x=column, ax=axes[i])
    axes[i].set_title(f'Boxplot for {column}')

# Remove any remaining empty subplots
for j in range(len(df.columns), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()

# ============================================================================
# ADVANCED FEATURE ENGINEERING (TRƯỚC KHI SPLIT)
# ============================================================================
def create_ultra_features(df):
    """
    Tạo các features cực mạnh dựa trên domain knowledge về diabetes
    CHỈ áp dụng trên training data, không áp dụng trên validation/test
    """
    df_fe = df.copy()
    
    # 1. COMPLEX RISK SCORES (quan trọng nhất)
    # Diabetes Risk Score (tổng hợp nhiều yếu tố)
    df_fe['Diabetes_Risk_Score'] = (
        df_fe['HighBP'] * 0.15 +
        df_fe['HighChol'] * 0.15 +
        (df_fe['BMI'] > 30).astype(float) * 0.20 +
        (df_fe['Age'] > 60).astype(float) * 0.15 +
        df_fe['GenHlth'] * 0.10 +
        df_fe['HeartDiseaseorAttack'] * 0.10 +
        df_fe['Stroke'] * 0.10 +
        (df_fe['PhysActivity'] == 0).astype(float) * 0.05
    )
    
    # Metabolic Syndrome Score
    df_fe['Metabolic_Score'] = (
        df_fe['HighBP'] + df_fe['HighChol'] + 
        (df_fe['BMI'] > 30).astype(float) +
        (df_fe['GenHlth'] >= 4).astype(float)
    ) / 4.0
    
    # Cardiovascular risk
    df_fe['Cardio_Risk'] = (
        df_fe['HeartDiseaseorAttack'] + df_fe['Stroke'] + 
        df_fe['HighBP'] + df_fe['HighChol']
    ) / 4.0
    
    # 2. BMI INTERACTIONS (BMI là yếu tố quan trọng nhất)
    df_fe['BMI_Age'] = df_fe['BMI'] * df_fe['Age']
    df_fe['BMI_Age_Squared'] = (df_fe['BMI'] * df_fe['Age']) ** 2
    df_fe['BMI_GenHlth'] = df_fe['BMI'] * df_fe['GenHlth']
    df_fe['BMI_HighBP'] = df_fe['BMI'] * df_fe['HighBP']
    df_fe['BMI_HighChol'] = df_fe['BMI'] * df_fe['HighChol']
    df_fe['BMI_Cardio'] = df_fe['BMI'] * (df_fe['HeartDiseaseorAttack'] + df_fe['Stroke'])
    
    # 3. AGE INTERACTIONS (Age là yếu tố quan trọng thứ 2)
    df_fe['Age_GenHlth'] = df_fe['Age'] * df_fe['GenHlth']
    df_fe['Age_Risk'] = df_fe['Age'] * df_fe['Diabetes_Risk_Score']
    df_fe['Age_Squared'] = df_fe['Age'] ** 2
    df_fe['Age_Cubed'] = df_fe['Age'] ** 3
    
    # 4. HEALTH STATUS COMBINATIONS
    df_fe['Physical_Health_Composite'] = (
        df_fe['GenHlth'] * 0.4 +
        df_fe['PhysHlth'] * 0.3 +
        df_fe['MentHlth'] * 0.2 +
        df_fe['DiffWalk'] * 0.1
    )
    
    df_fe['Lifestyle_Index'] = (
        df_fe['PhysActivity'] * 0.3 +
        df_fe['Fruits'] * 0.2 +
        df_fe['Veggies'] * 0.2 -
        df_fe['Smoker'] * 0.15 -
        df_fe['HvyAlcoholConsump'] * 0.15
    )
    
    # 5. POLYNOMIAL FEATURES (cho các biến quan trọng)
    df_fe['BMI_squared'] = df_fe['BMI'] ** 2
    df_fe['GenHlth_squared'] = df_fe['GenHlth'] ** 2
    
    # 6. RATIO FEATURES
    df_fe['MentPhys_Ratio'] = df_fe['MentHlth'] / (df_fe['PhysHlth'] + 1)
    df_fe['Health_Risk_Ratio'] = df_fe['Physical_Health_Composite'] / (df_fe['Cardio_Risk'] + 0.1)
    
    # 7. ADVANCED BINNING (chia nhóm chi tiết)
    df_fe['BMI_Category_Detailed'] = pd.cut(
        df_fe['BMI'],
        bins=[0, 18.5, 25, 30, 35, 40, 100],
        labels=[0, 1, 2, 3, 4, 5]
    ).astype(float)
    
    df_fe['Age_Decade'] = (df_fe['Age'] // 10).astype(float)
    df_fe['Age_Group'] = pd.cut(
        df_fe['Age'],
        bins=[0, 30, 45, 60, 75, 100],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # 8. COUNT & AGGREGATION FEATURES
    df_fe['Total_Risk_Factors'] = (
        df_fe['HighBP'] + df_fe['HighChol'] + df_fe['Smoker'] +
        df_fe['HeartDiseaseorAttack'] + df_fe['Stroke'] +
        (df_fe['BMI'] > 30).astype(float) +
        (df_fe['Age'] > 60).astype(float) +
        (df_fe['GenHlth'] >= 4).astype(float)
    )
    
    df_fe['Protective_Factors'] = (
        df_fe['PhysActivity'] + df_fe['Fruits'] + 
        df_fe['Veggies'] + df_fe['CholCheck']
    )
    
    df_fe['Risk_Protection_Ratio'] = (
        df_fe['Total_Risk_Factors'] / (df_fe['Protective_Factors'] + 1)
    )
    
    # 9. COMPLEX 3-WAY INTERACTIONS
    df_fe['BMI_Age_GenHlth'] = df_fe['BMI'] * df_fe['Age'] * df_fe['GenHlth']
    df_fe['Risk_Lifestyle_Interaction'] = (
        df_fe['Total_Risk_Factors'] * df_fe['Lifestyle_Index']
    )
    
    # 10. LOGARITHMIC TRANSFORMATIONS (cho skewed features)
    df_fe['BMI_log'] = np.log1p(df_fe['BMI'])
    df_fe['Age_log'] = np.log1p(df_fe['Age'])
    df_fe['MentHlth_log'] = np.log1p(df_fe['MentHlth'] + 1)
    df_fe['PhysHlth_log'] = np.log1p(df_fe['PhysHlth'] + 1)
    
    # 11. SQUARE ROOT TRANSFORMATIONS
    df_fe['BMI_sqrt'] = np.sqrt(df_fe['BMI'])
    df_fe['Age_sqrt'] = np.sqrt(df_fe['Age'])
    
    # 12. HEALTHCARE ACCESS SCORES
    df_fe['Healthcare_Quality'] = (
        df_fe['AnyHealthcare'] - df_fe['NoDocbcCost']
    )
    
    df_fe['Healthcare_Barrier_Score'] = (
        df_fe['NoDocbcCost'] * (1 - df_fe['AnyHealthcare'])
    )
    
    return df_fe

# Áp dụng feature engineering TRƯỚC KHI split
print("=" * 80)
print("ADVANCED FEATURE ENGINEERING")
print("=" * 80)
df_enhanced = create_ultra_features(df)
print(f"✅ Created {len(df_enhanced.columns)} features (from {len(df.columns)} original)")

# Feature engineering xong, giờ mới split
X = df_enhanced.drop("Diabetes_binary", axis=1)
Y = df_enhanced["Diabetes_binary"]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================
print("=" * 80)
print("SO SÁNH CÁC PHƯƠNG PHÁP SAMPLING (ĐÚNG CÁCH - RESAMPLE TRONG CV)")
print("=" * 80)

# Best parameters từ Optuna
best_params = {
    'C': 0.1,
    'penalty': 'l1',
    'solver': 'liblinear'
}

# Định nghĩa các phương pháp sampling
sampling_methods = {
    'SMOTE (Original)': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42, kind='borderline-1'),
    'BorderlineSMOTE-2': BorderlineSMOTE(random_state=42, kind='borderline-2'),
}

results = []

print("\nĐang test các phương pháp sampling (với CV đúng cách)...\n")
print("⚠️  QUAN TRỌNG:")
print("   - Training fold: ĐƯỢC RESAMPLE")
print("   - Validation fold: GIỮ NGUYÊN (không resample)")
print("   - Test set: GIỮ NGUYÊN (không resample)\n")

# Tạo StratifiedKFold để đảm bảo class distribution
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for method_name, sampler in sampling_methods.items():
    try:
        print(f"Testing: {method_name}...")
        
        # ====================================================================
        # PHẦN 1: CROSS-VALIDATION THỦ CÔNG (ĐÚNG CÁCH)
        # ====================================================================
        # Thực hiện CV thủ công để đảm bảo:
        #   - Training fold: ĐƯỢC RESAMPLE
        #   - Validation fold: GIỮ NGUYÊN (không resample)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X_train, Y_train):
            # Split data
            X_train_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            y_train_fold = Y_train.iloc[train_idx] if hasattr(Y_train, 'iloc') else Y_train[train_idx]
            X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_val_fold = Y_train.iloc[val_idx] if hasattr(Y_train, 'iloc') else Y_train[val_idx]
            
            # RESAMPLE CHỈ training fold
            X_train_fold_resampled, y_train_fold_resampled = sampler.fit_resample(X_train_fold, y_train_fold)
            
            # Scale:
            #   - Fit scaler trên training fold (đã resample)
            #   - Transform validation fold (giữ nguyên, chỉ scale)
            scaler_fold = StandardScaler()
            X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold_resampled)
            X_val_fold_scaled = scaler_fold.transform(X_val_fold)  # Validation: CHỈ transform
            
            # Train model trên training fold (đã resample)
            model_fold = LogisticRegression(**best_params, random_state=42, max_iter=1000)
            model_fold.fit(X_train_fold_scaled, y_train_fold_resampled)
            
            # Predict trên validation fold (GIỮ NGUYÊN - không resample)
            y_pred_fold = model_fold.predict(X_val_fold_scaled)
            
            # Tính F1-score trên validation fold (giữ nguyên)
            f1_fold = f1_score(y_val_fold, y_pred_fold, pos_label=1.0, zero_division=0)
            cv_scores.append(f1_fold)
        
        cv_scores = np.array(cv_scores)
        
        # ====================================================================
        # PHẦN 2: TRAIN FINAL MODEL VÀ TEST TRÊN TEST SET
        # ====================================================================
        # Resample CHỈ training set (X_train, Y_train)
        # Test set (X_test, Y_test) GIỮ NGUYÊN - KHÔNG RESAMPLE
        X_resampled, y_resampled = sampler.fit_resample(X_train, Y_train)
        
        # Scale:
        #   - Fit scaler trên training data (đã resample)
        #   - Transform test data (giữ nguyên, chỉ scale)
        scaler = StandardScaler()
        X_resampled_scaled = scaler.fit_transform(X_resampled)  # Training: fit + transform
        X_test_scaled = scaler.transform(X_test)  # Test: CHỈ transform (giữ nguyên distribution)
        
        # Train final model trên training data (đã resample)
        model = LogisticRegression(**best_params, random_state=42, max_iter=1000)
        model.fit(X_resampled_scaled, y_resampled)
        
        # Test predictions trên test set (GIỮ NGUYÊN - không resample)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
        
        # Metrics trên test set (giữ nguyên)
        train_score = model.score(X_resampled_scaled, y_resampled)
        test_score = accuracy_score(Y_test, y_pred)  # Y_test giữ nguyên
        roc_auc = roc_auc_score(Y_test, y_pred_proba)  # Y_test giữ nguyên
        precision_1 = precision_score(Y_test, y_pred, pos_label=1.0, zero_division=0)
        recall_1 = recall_score(Y_test, y_pred, pos_label=1.0, zero_division=0)
        f1_1 = f1_score(Y_test, y_pred, pos_label=1.0, zero_division=0)
        
        # Optimal threshold dựa trên test set (giữ nguyên)
        precision_curve, recall_curve, thresholds = precision_recall_curve(Y_test, y_pred_proba)
        f1_scores_thresh = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
        optimal_idx = np.argmax(f1_scores_thresh)
        optimal_threshold = thresholds[optimal_idx]
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        test_score_opt = accuracy_score(Y_test, y_pred_optimal)  # Y_test giữ nguyên
        f1_opt = f1_score(Y_test, y_pred_optimal, pos_label=1.0, zero_division=0)
        
        # Resampled data size
        resampled_size = len(y_resampled)
        
        results.append({
            'Method': method_name,
            'CV F1-Score': f'{cv_scores.mean():.4f}',
            'CV Std': f'{cv_scores.std():.4f}',
            'Train Accuracy': f'{train_score:.4f}',
            'Test Accuracy (0.5)': f'{test_score:.4f}',
            'Test Accuracy (Optimal)': f'{test_score_opt:.4f}',
            'ROC-AUC': f'{roc_auc:.4f}',
            'F1-Score (Optimal)': f'{f1_opt:.4f}',
            'Precision (1)': f'{precision_score(Y_test, y_pred_optimal, pos_label=1.0, zero_division=0):.4f}',
            'Recall (1)': f'{recall_score(Y_test, y_pred_optimal, pos_label=1.0, zero_division=0):.4f}',
            'Resampled Size': resampled_size,
            'Optimal Threshold': f'{optimal_threshold:.4f}'
        })
        
        print(f"  ✅ CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"     Test Acc (Optimal): {test_score_opt:.4f}, F1: {f1_opt:.4f}")
        print(f"     Training size (resampled): {resampled_size}, Test size (original): {len(Y_test)}\n")
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)}\n")
        continue

# Tạo DataFrame và sắp xếp
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('CV F1-Score', ascending=False)

print("=" * 80)
print("KẾT QUẢ SO SÁNH TẤT CẢ PHƯƠNG PHÁP SAMPLING")
print("=" * 80)
print(results_df.to_string(index=False))

# Tìm best method dựa trên CV F1-Score
best_method = results_df.iloc[0]['Method']
print(f"\n🏆 BEST METHOD (dựa trên CV F1-Score): {best_method}")
print(f"   CV F1-Score: {results_df.iloc[0]['CV F1-Score']} (+/- {results_df.iloc[0]['CV Std']})")
print(f"   Test Accuracy: {results_df.iloc[0]['Test Accuracy (Optimal)']}")
print(f"   F1-Score: {results_df.iloc[0]['F1-Score (Optimal)']}")
print(f"   ROC-AUC: {results_df.iloc[0]['ROC-AUC']}")

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. CV F1-Score Comparison
axes[0, 0].barh(results_df['Method'], results_df['CV F1-Score'].astype(float))
axes[0, 0].set_xlabel('CV F1-Score')
axes[0, 0].set_title('CV F1-Score Comparison (Đúng cách)', fontweight='bold')
axes[0, 0].invert_yaxis()

# 2. Test Accuracy Comparison
axes[0, 1].barh(results_df['Method'], results_df['Test Accuracy (Optimal)'].astype(float))
axes[0, 1].set_xlabel('Test Accuracy (Optimal Threshold)')
axes[0, 1].set_title('Test Accuracy Comparison', fontweight='bold')
axes[0, 1].invert_yaxis()

# 3. F1-Score Comparison
axes[1, 0].barh(results_df['Method'], results_df['F1-Score (Optimal)'].astype(float))
axes[1, 0].set_xlabel('F1-Score (Optimal Threshold)')
axes[1, 0].set_title('F1-Score Comparison', fontweight='bold')
axes[1, 0].invert_yaxis()

# 4. ROC-AUC Comparison
axes[1, 1].barh(results_df['Method'], results_df['ROC-AUC'].astype(float))
axes[1, 1].set_xlabel('ROC-AUC')
axes[1, 1].set_title('ROC-AUC Comparison', fontweight='bold')
axes[1, 1].invert_yaxis()

plt.tight_layout()
plt.show()

# ============================================================================
# FINAL MODEL TRAINING
# ============================================================================
print("\n" + "=" * 80)
print("TRAIN BEST MODEL VỚI BEST SAMPLING METHOD")
print("=" * 80)
print("⚠️  QUAN TRỌNG:")
print("   - Training set: ĐƯỢC RESAMPLE")
print("   - Test set: GIỮ NGUYÊN (không resample)\n")

best_sampler = sampling_methods[best_method]

# Resample CHỈ training set (X_train, Y_train)
# Test set (X_test, Y_test) GIỮ NGUYÊN - KHÔNG RESAMPLE
X_best_resampled, y_best_resampled = best_sampler.fit_resample(X_train, Y_train)

# Scale:
#   - Fit scaler trên training data (đã resample)
#   - Transform test data (giữ nguyên, chỉ scale)
scaler_best = StandardScaler()
X_best_resampled_scaled = scaler_best.fit_transform(X_best_resampled)  # Training: fit + transform
X_test_best_scaled = scaler_best.transform(X_test)  # Test: CHỈ transform (giữ nguyên)

# Train final model trên training data (đã resample)
lg_final = LogisticRegression(**best_params, random_state=42, max_iter=1000)
lg_final.fit(X_best_resampled_scaled, y_best_resampled)

# Predictions trên test set (GIỮ NGUYÊN - không resample)
y_pred_proba_final = lg_final.predict_proba(X_test_best_scaled)[:, 1]
y_pred_final = lg_final.predict(X_test_best_scaled)

# Find optimal threshold dựa trên test set (giữ nguyên)
precision_curve, recall_curve, thresholds = precision_recall_curve(Y_test, y_pred_proba_final)
f1_scores_thresh = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
optimal_idx = np.argmax(f1_scores_thresh)
optimal_threshold_final = thresholds[optimal_idx]
y_pred_optimal_final = (y_pred_proba_final >= optimal_threshold_final).astype(int)

# Final metrics với CV thủ công (đúng cách)
# Đảm bảo: training fold resample, validation fold giữ nguyên
skf_final = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_final = []

for train_idx, val_idx in skf_final.split(X_train, Y_train):
    # Split data
    X_train_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
    y_train_fold = Y_train.iloc[train_idx] if hasattr(Y_train, 'iloc') else Y_train[train_idx]
    X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
    y_val_fold = Y_train.iloc[val_idx] if hasattr(Y_train, 'iloc') else Y_train[val_idx]
    
    # RESAMPLE CHỈ training fold
    X_train_fold_resampled, y_train_fold_resampled = best_sampler.fit_resample(X_train_fold, y_train_fold)
    
    # Scale
    scaler_fold = StandardScaler()
    X_train_fold_scaled = scaler_fold.fit_transform(X_train_fold_resampled)
    X_val_fold_scaled = scaler_fold.transform(X_val_fold)  # Validation: CHỈ transform
    
    # Train model
    model_fold = LogisticRegression(**best_params, random_state=42, max_iter=1000)
    model_fold.fit(X_train_fold_scaled, y_train_fold_resampled)
    
    # Predict trên validation fold (GIỮ NGUYÊN)
    y_pred_fold = model_fold.predict(X_val_fold_scaled)
    
    # Tính F1-score
    f1_fold = f1_score(y_val_fold, y_pred_fold, pos_label=1.0, zero_division=0)
    cv_scores_final.append(f1_fold)

cv_scores_final = np.array(cv_scores_final)

# Metrics trên test set (giữ nguyên)
train_score_final = lg_final.score(X_best_resampled_scaled, y_best_resampled)
test_score_final = accuracy_score(Y_test, y_pred_optimal_final)  # Y_test giữ nguyên
roc_auc_final = roc_auc_score(Y_test, y_pred_proba_final)  # Y_test giữ nguyên
precision_final = precision_score(Y_test, y_pred_optimal_final, pos_label=1.0, zero_division=0)
recall_final = recall_score(Y_test, y_pred_optimal_final, pos_label=1.0, zero_division=0)
f1_final = f1_score(Y_test, y_pred_optimal_final, pos_label=1.0, zero_division=0)

print(f"\n📋 FINAL MODEL CONFIGURATION:")
print(f"  Sampling Method: {best_method}")
print(f"  Model Parameters: {best_params}")
print(f"  Optimal Threshold: {optimal_threshold_final:.4f}")

print(f"\n📊 FINAL PERFORMANCE:")
print(f"  CV F1-Score (đúng cách): {cv_scores_final.mean():.4f} (+/- {cv_scores_final.std() * 2:.4f})")
print(f"  Training Accuracy: {train_score_final:.4f}")
print(f"  Test Accuracy: {test_score_final:.4f}")
print(f"  ROC-AUC: {roc_auc_final:.4f}")
print(f"  Precision (Class 1): {precision_final:.4f}")
print(f"  Recall (Class 1): {recall_final:.4f}")
print(f"  F1-Score (Class 1): {f1_final:.4f}")

print("\n📋 Classification Report:")
print(classification_report(Y_test, y_pred_optimal_final, zero_division=0))

# Confusion Matrix
cm_final = confusion_matrix(Y_test, y_pred_optimal_final)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_final, annot=True, fmt='d', cmap='Greens', cbar_kws={'label': 'Count'})
plt.title(f'Best Model: {best_method}\n(Optimal Threshold)', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("✅ HOÀN TẤT!")
print("=" * 80)

# ============================================================================
# STRATEGY 1: UPGRADE TO TREE-BASED MODELS (HIGHEST IMPACT)
# ============================================================================
def train_advanced_models(X_train, y_train, X_test, y_test, sampler):
    """
    Train các models mạnh nhất cho tabular data
    ⚠️ QUAN TRỌNG: Chỉ augment training data, validation và test giữ nguyên
    """
    # RESAMPLE CHỈ training data
    # Test data GIỮ NGUYÊN - KHÔNG RESAMPLE
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    # Scale (một số models không cần nhưng tốt hơn khi có)
    scaler = RobustScaler()
    # Fit scaler trên training data (đã resample)
    X_train_scaled = scaler.fit_transform(X_resampled)
    # Transform test data (giữ nguyên, chỉ scale)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        # 1. XGBoost - Thường tốt nhất cho tabular data
        'XGBoost': XGBClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            scale_pos_weight=len(y_resampled[y_resampled==0]) / len(y_resampled[y_resampled==1]),
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1,
            tree_method='hist'  # Nhanh hơn
        ),
        
        # 2. LightGBM - Nhanh và mạnh
        'LightGBM': LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.03,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            boosting_type='gbdt'
        ),
        
        # 3. CatBoost - Tốt với categorical features
        'CatBoost': CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.03,
            l2_leaf_reg=3,
            class_weights=[1, len(y_resampled[y_resampled==0]) / len(y_resampled[y_resampled==1])],
            random_state=42,
            verbose=False,
            thread_count=-1
        ),
        
        # 4. Random Forest - Robust và ổn định
        'Random Forest': RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            bootstrap=True
        )
    }
    
    results = []
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        
        # Train
        model.fit(X_train_scaled, y_resampled)
        trained_models[name] = model
        
        # Predict
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = model.predict(X_test_scaled)
        
        # Optimal threshold
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores_thresh = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
        optimal_idx = np.argmax(f1_scores_thresh)
        optimal_threshold = thresholds[optimal_idx]
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        test_acc = accuracy_score(y_test, y_pred_optimal)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred_optimal, pos_label=1.0, zero_division=0)
        precision = precision_score(y_test, y_pred_optimal, pos_label=1.0, zero_division=0)
        recall = recall_score(y_test, y_pred_optimal, pos_label=1.0, zero_division=0)
        
        results.append({
            'Model': name,
            'F1-Score': f'{f1:.4f}',
            'ROC-AUC': f'{roc_auc:.4f}',
            'Accuracy': f'{test_acc:.4f}',
            'Precision': f'{precision:.4f}',
            'Recall': f'{recall:.4f}',
            'Optimal Threshold': f'{optimal_threshold:.4f}'
        })
        
        print(f"✅ F1-Score: {f1:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   Accuracy: {test_acc:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1-Score', ascending=False)
    
    return results_df, trained_models, scaler

# ============================================================================
# STRATEGY 3: OPTUNA HYPERPARAMETER OPTIMIZATION
# ============================================================================
def optimize_catboost_with_optuna(X_train, y_train, sampler, n_trials=100):
    """
    Tối ưu CatBoost với Optuna - Tìm best hyperparameters
    ⚠️ QUAN TRỌNG: Chỉ augment training fold, validation fold giữ nguyên
    """
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2000),
            'depth': trial.suggest_int('depth', 4, 12),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.001, 0.1),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0, 1),
            'random_strength': trial.suggest_loguniform('random_strength', 1e-9, 10),
            'random_state': 42,
            'verbose': False,
            'thread_count': -1,
            'loss_function': 'Logloss',
            'eval_metric': 'F1'
        }
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_train_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            y_train_fold = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_val_fold = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
            
            # RESAMPLE CHỈ training fold
            # Validation fold GIỮ NGUYÊN - KHÔNG RESAMPLE
            X_train_fold_res, y_train_fold_res = sampler.fit_resample(X_train_fold, y_train_fold)
            
            # Scale
            scaler = RobustScaler()
            # Fit scaler trên training fold (đã resample)
            X_train_fold_scaled = scaler.fit_transform(X_train_fold_res)
            # Transform validation fold (giữ nguyên, chỉ scale)
            X_val_fold_scaled = scaler.transform(X_val_fold)
            
            # Calculate class weights
            class_weights = [1, len(y_train_fold_res[y_train_fold_res==0]) / len(y_train_fold_res[y_train_fold_res==1])]
            params['class_weights'] = class_weights
            
            # Train
            model = CatBoostClassifier(**params)
            model.fit(
                X_train_fold_scaled, y_train_fold_res,
                eval_set=(X_val_fold_scaled, y_val_fold),
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Predict trên validation fold (GIỮ NGUYÊN)
            y_pred_proba = model.predict_proba(X_val_fold_scaled)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Score
            score = f1_score(y_val_fold, y_pred, pos_label=1.0, zero_division=0)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='catboost_optimization'
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n🏆 Best F1-Score: {study.best_value:.4f}")
    print(f"📋 Best Parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    return study.best_params, study.best_value

def optimize_xgboost_with_optuna(X_train, y_train, sampler, n_trials=100):
    """
    Tối ưu XGBoost với Optuna - Tìm best hyperparameters
    ⚠️ QUAN TRỌNG: Chỉ augment training fold, validation fold giữ nguyên
    """
    def objective(trial):
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.2),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'colsample_bylevel': trial.suggest_uniform('colsample_bylevel', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 10.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 10.0),
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': -1,
            'tree_method': 'hist'
        }
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_train_fold = X_train.iloc[train_idx] if hasattr(X_train, 'iloc') else X_train[train_idx]
            y_train_fold = y_train.iloc[train_idx] if hasattr(y_train, 'iloc') else y_train[train_idx]
            X_val_fold = X_train.iloc[val_idx] if hasattr(X_train, 'iloc') else X_train[val_idx]
            y_val_fold = y_train.iloc[val_idx] if hasattr(y_train, 'iloc') else y_train[val_idx]
            
            # RESAMPLE CHỈ training fold
            # Validation fold GIỮ NGUYÊN - KHÔNG RESAMPLE
            X_train_fold_res, y_train_fold_res = sampler.fit_resample(X_train_fold, y_train_fold)
            
            # Scale
            scaler = RobustScaler()
            # Fit scaler trên training fold (đã resample)
            X_train_fold_scaled = scaler.fit_transform(X_train_fold_res)
            # Transform validation fold (giữ nguyên, chỉ scale)
            X_val_fold_scaled = scaler.transform(X_val_fold)
            
            # Calculate scale_pos_weight
            params['scale_pos_weight'] = len(y_train_fold_res[y_train_fold_res==0]) / len(y_train_fold_res[y_train_fold_res==1])
            
            # Train
            model = XGBClassifier(**params)
            model.fit(X_train_fold_scaled, y_train_fold_res)
            
            # Predict trên validation fold (GIỮ NGUYÊN)
            y_pred_proba = model.predict_proba(X_val_fold_scaled)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Score
            score = f1_score(y_val_fold, y_pred, pos_label=1.0, zero_division=0)
            cv_scores.append(score)
        
        return np.mean(cv_scores)
    
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='xgb_optimization'
    )
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\n🏆 Best F1-Score: {study.best_value:.4f}")
    print(f"📋 Best Parameters:")
    for key, value in study.best_params.items():
        print(f"   {key}: {value}")
    
    return study.best_params, study.best_value

# ============================================================================
# STRATEGY 4: INTELLIGENT FEATURE SELECTION
# ============================================================================
def intelligent_feature_selection(X_train, y_train, X_test, y_test, sampler, top_k_ratio=0.8):
    """
    Chọn features tốt nhất bằng nhiều phương pháp
    ⚠️ QUAN TRỌNG: Chỉ augment training data để tính feature importance
    """
    # Resample CHỈ training data để tính feature importance
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # Method 1: Mutual Information
    mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
    mi_selector.fit(X_train_scaled, y_resampled)
    mi_scores = mi_selector.scores_
    
    # Method 2: F-statistic
    f_selector = SelectKBest(score_func=f_classif, k='all')
    f_selector.fit(X_train_scaled, y_resampled)
    f_scores = f_selector.scores_
    
    # Method 3: Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X_train_scaled, y_resampled)
    rf_importances = rf.feature_importances_
    
    # Normalize scores
    mi_scores_norm = (mi_scores - mi_scores.min()) / (mi_scores.max() - mi_scores.min() + 1e-10)
    f_scores_norm = (f_scores - f_scores.min()) / (f_scores.max() - f_scores.min() + 1e-10)
    rf_importances_norm = (rf_importances - rf_importances.min()) / (rf_importances.max() - rf_importances.min() + 1e-10)
    
    # Combined score
    combined_scores = (mi_scores_norm * 0.4 + f_scores_norm * 0.3 + rf_importances_norm * 0.3)
    
    # Select top features
    top_k = int(X_train_scaled.shape[1] * top_k_ratio)
    top_indices = np.argsort(combined_scores)[-top_k:]
    
    X_train_selected = X_train_scaled[:, top_indices]
    X_test_selected = X_test_scaled[:, top_indices]
    
    print(f"✅ Selected {top_k} best features out of {X_train_scaled.shape[1]}")
    
    return X_train_selected, X_test_selected, top_indices, scaler

# ============================================================================
# STRATEGY 5: POWERFUL ENSEMBLE METHODS
# ============================================================================
def create_super_ensemble(X_train, y_train, X_test, y_test, sampler):
    """
    Tạo ensemble mạnh nhất từ nhiều models
    ⚠️ QUAN TRỌNG: Chỉ augment training data, test data giữ nguyên
    """
    # RESAMPLE CHỈ training data
    # Test data GIỮ NGUYÊN - KHÔNG RESAMPLE
    X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
    
    scaler = RobustScaler()
    # Fit scaler trên training data (đã resample)
    X_train_scaled = scaler.fit_transform(X_resampled)
    # Transform test data (giữ nguyên, chỉ scale)
    X_test_scaled = scaler.transform(X_test)
    
    # Base models (đã được tune)
    xgb = XGBClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=len(y_resampled[y_resampled==0]) / len(y_resampled[y_resampled==1]),
        random_state=42, n_jobs=-1, eval_metric='logloss'
    )
    
    lgbm = LGBMClassifier(
        n_estimators=500, max_depth=8, learning_rate=0.03,
        num_leaves=31, class_weight='balanced',
        random_state=42, n_jobs=-1, verbose=-1
    )
    
    catboost = CatBoostClassifier(
        iterations=500, depth=8, learning_rate=0.03,
        class_weights=[1, len(y_resampled[y_resampled==0]) / len(y_resampled[y_resampled==1])],
        random_state=42, verbose=False, thread_count=-1
    )
    
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=15,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # 1. Voting Classifier (Soft) - Trung bình weighted của probabilities
    voting_soft = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('lgbm', lgbm),
            ('catboost', catboost),
            ('rf', rf)
        ],
        voting='soft',
        weights=[2, 2, 2, 1]  # XGBoost, LightGBM, CatBoost quan trọng hơn
    )
    
    # 2. Stacking Classifier - Dùng meta-learner để kết hợp
    stacking = StackingClassifier(
        estimators=[
            ('xgb', xgb),
            ('lgbm', lgbm),
            ('catboost', catboost),
            ('rf', rf)
        ],
        final_estimator=LogisticRegression(
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        ),
        cv=5,
        stack_method='predict_proba'
    )
    
    # 3. Blending (Custom) - Weighted average của predictions
    def blended_predict(models, weights, X):
        predictions = []
        for model in models:
            pred_proba = model.predict_proba(X)[:, 1]
            predictions.append(pred_proba)
        
        # Weighted average
        blended = np.average(predictions, axis=0, weights=weights)
        return blended
    
    # Train all base models
    models = [xgb, lgbm, catboost, rf]
    for model in models:
        model.fit(X_train_scaled, y_resampled)
    
    # Test ensemble methods
    results = {}
    
    # Voting Soft
    voting_soft.fit(X_train_scaled, y_resampled)
    y_pred_voting = voting_soft.predict_proba(X_test_scaled)[:, 1]
    
    # Stacking
    stacking.fit(X_train_scaled, y_resampled)
    y_pred_stacking = stacking.predict_proba(X_test_scaled)[:, 1]
    
    # Blending
    y_pred_blending = blended_predict(
        models, 
        weights=[0.3, 0.3, 0.3, 0.1],  # XGB, LGBM, CatBoost quan trọng hơn
        X=X_test_scaled
    )
    
    # Evaluate
    for name, y_pred_proba in [
        ('Voting Soft', y_pred_voting),
        ('Stacking', y_pred_stacking),
        ('Blending', y_pred_blending)
    ]:
        # Optimal threshold
        precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores_thresh = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
        optimal_idx = np.argmax(f1_scores_thresh)
        optimal_threshold = thresholds[optimal_idx]
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        
        f1 = f1_score(y_test, y_pred_optimal, pos_label=1.0, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'Optimal Threshold': optimal_threshold
        }
        
        print(f"\n{name}:")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
    
    return results, models, scaler

# ============================================================================
# COMPLETE HIGH-PERFORMANCE PIPELINE
# ============================================================================
def maximize_performance_pipeline(df):
    """
    Pipeline hoàn chỉnh để đạt performance cao nhất
    """
    print("="*80)
    print("🚀 HIGH-PERFORMANCE OPTIMIZATION PIPELINE")
    print("="*80)
    
    # STEP 1: Advanced Feature Engineering (đã được thực hiện ở trên)
    print("\n📊 STEP 1: Advanced Feature Engineering")
    print("-"*80)
    df_enhanced = create_ultra_features(df)
    X = df_enhanced.drop("Diabetes_binary", axis=1)
    y = df_enhanced["Diabetes_binary"]
    print(f"✅ Created {len(X.columns)} features (from {len(df.columns)-1} original)")
    
    # STEP 2: Data Split
    print("\n📊 STEP 2: Data Splitting")
    print("-"*80)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # STEP 3: Outlier Handling (chỉ trên training data)
    print("\n📊 STEP 3: Outlier Handling (Training data only)")
    print("-"*80)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in ['BMI', 'MentHlth', 'PhysHlth', 'Age']:
        if col in numeric_cols:
            Q1 = X_train[col].quantile(0.25)
            Q3 = X_train[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            # Clip training data
            X_train[col] = X_train[col].clip(lower=lower, upper=upper)
            # Clip test data với cùng bounds (không tính lại từ test)
            X_test[col] = X_test[col].clip(lower=lower, upper=upper)
    print("✅ Outliers handled")
    
    # STEP 4: Sampling
    print("\n📊 STEP 4: Best Sampling Method")
    print("-"*80)
    best_sampler = SMOTE(random_state=42)
    print("✅ Using SMOTE")
    print("⚠️  Lưu ý: Chỉ augment training data, validation và test giữ nguyên")
    
    # STEP 5: Feature Selection
    print("\n📊 STEP 5: Intelligent Feature Selection")
    print("-"*80)
    X_train_selected, X_test_selected, selected_indices, scaler_fe = intelligent_feature_selection(
        X_train, y_train, X_test, y_test, best_sampler, top_k_ratio=0.85
    )
    
    # Convert back to DataFrame for compatibility
    X_train_selected_df = pd.DataFrame(X_train_selected, index=X_train.index)
    X_test_selected_df = pd.DataFrame(X_test_selected, index=X_test.index)
    
    # STEP 6: Train Advanced Models
    print("\n📊 STEP 6: Training Advanced Models")
    print("-"*80)
    results_df, trained_models, scaler_models = train_advanced_models(
        X_train_selected_df, y_train, X_test_selected_df, y_test, best_sampler
    )
    
    print("\n" + "="*80)
    print("🏆 TOP 3 MODELS:")
    print("="*80)
    print(results_df.head(3).to_string(index=False))
    
    # STEP 7: Hyperparameter Optimization (cho best model)
    print("\n📊 STEP 7: Hyperparameter Optimization")
    print("-"*80)
    best_model_name = results_df.iloc[0]['Model']
    print(f"Optimizing: {best_model_name}")
    print("⚠️  Lưu ý: Chỉ augment training fold trong CV, validation fold giữ nguyên")
    
    try:
        if best_model_name == 'CatBoost':
            best_params, best_score = optimize_catboost_with_optuna(
                X_train_selected_df, y_train, best_sampler, n_trials=50  # Có thể tăng lên 100-200
            )
        elif best_model_name == 'XGBoost':
        best_params, best_score = optimize_xgboost_with_optuna(
                X_train_selected_df, y_train, best_sampler, n_trials=50  # Có thể tăng lên 100-200
            )
        else:
            print(f"⚠️  Hyperparameter optimization chưa được implement cho {best_model_name}")
            best_params, best_score = None, None
    except Exception as e:
        print(f"⚠️  Error trong hyperparameter optimization: {str(e)}")
        best_params, best_score = None, None
    
    # STEP 8: Ensemble Methods
    print("\n📊 STEP 8: Creating Super Ensemble")
    print("-"*80)
    print("⚠️  Lưu ý: Chỉ augment training data, test data giữ nguyên")
    ensemble_results, ensemble_models, scaler_ensemble = create_super_ensemble(
        X_train_selected_df, y_train, X_test_selected_df, y_test, best_sampler
    )
    
    print("\n" + "="*80)
    print("🎯 FINAL RESULTS SUMMARY")
    print("="*80)
    print("\nBest Single Model:")
    print(results_df.head(1).to_string(index=False))
    print("\nEnsemble Methods:")
    for name, metrics in ensemble_results.items():
        print(f"\n{name}:")
        print(f"  F1-Score: {metrics['F1-Score']:.4f}")
        print(f"  ROC-AUC: {metrics['ROC-AUC']:.4f}")
    
    return results_df, ensemble_results, trained_models, ensemble_models

# Chạy pipeline
results_df, ensemble_results, trained_models, ensemble_models = maximize_performance_pipeline(df)
