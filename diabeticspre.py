import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading dataset...")
data = pd.read_csv(r"c:\Users\Abina\py\dbequal.csv")

print("Data shape:", data.shape)
print("Class distribution:")
print(data['Diabetes_binary'].value_counts())

# Advanced Feature Engineering
def create_advanced_features(df):
    df_new = df.copy()

    df_new['BMI_category'] = pd.cut(
        df_new['BMI'],
        bins=[0, 18.5, 24.9, 29.9, float('inf')],
        labels=[0, 1, 2, 3]
    ).astype(int)  # Convert to int

    df_new['Age_BMI_interaction'] = df_new['Age'] * df_new['BMI']

    df_new['Health_Risk_Score'] = (
        df_new['HighBP'] * 2 +
        df_new['HighChol'] * 2 +
        df_new['BMI']/10 +
        df_new['Smoker'] +
        df_new['HeartDiseaseorAttack'] * 3 +
        df_new['GenHlth'] +
        df_new['Age']/10
    )

    df_new['PhysActivity_Health'] = df_new['PhysActivity'] * (6 - df_new['GenHlth'])

    df_new['Lifestyle_Risk'] = (
        df_new['Smoker'] +
        df_new['HvyAlcoholConsump'] +
        (1 - df_new['PhysActivity']) +
        (1 - df_new['Fruits']) +
        (1 - df_new['Veggies'])
    )

    df_new['Age_Group'] = pd.cut(
        df_new['Age'],
        bins=[0, 3, 7, 11, 13],
        labels=[0, 1, 2, 3]
    ).astype(int)  # Convert to int

    df_new['Mental_Physical_Health'] = df_new['MentHlth'] + df_new['PhysHlth']

    df_new['Healthcare_Risk'] = (
        (1 - df_new['AnyHealthcare']) +
        df_new['NoDocbcCost'] +
        (1 - df_new['CholCheck'])
    )

    df_new['Sex_Age_interaction'] = df_new['Sex'] * df_new['Age']

    df_new['Cardio_Risk'] = (
        df_new['HighBP'] +
        df_new['HighChol'] +
        df_new['HeartDiseaseorAttack'] +
        df_new['Stroke']
    )

    return df_new

data_enhanced = create_advanced_features(data)

X = data_enhanced.drop("Diabetes_binary", axis=1)
y = data_enhanced["Diabetes_binary"]

print(f"\nEnhanced feature count: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Feature selection
print("\nPerforming feature selection...")
selector = SelectKBest(score_func=f_classif, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X.columns[selector.get_support()].tolist()
print("Selected features:", selected_features)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)

# SMOTE
print("\nApplying SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)

# Model definitions
models = {
    'XGBoost': XGBClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_weight=3,
        gamma=0.1,
        eval_metric='logloss',
        random_state=42
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=800,
        max_depth=8,
        learning_rate=0.03,
        subsample=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        min_child_samples=5,
        num_leaves=50,
        random_state=42,
        verbose=1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=800,
        depth=8,
        learning_rate=0.03,
        l2_leaf_reg=3,
        border_count=128,
        random_state=42,
        verbose=False
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=800,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
}

# Training individual models
print("\nTraining individual models...")
trained_models = {}
individual_scores = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_balanced, y_train_balanced)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    individual_scores[name] = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'f1_score': f1
    }
    trained_models[name] = model

    print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}")

# Ensemble models
print("\nCreating ensemble models...")

voting_clf = VotingClassifier(
    estimators=[
        ('xgb', trained_models['XGBoost']),
        ('lgb', trained_models['LightGBM']),
        ('cb', trained_models['CatBoost']),
        ('rf', trained_models['Random Forest'])
    ],
    voting='soft'
)

voting_clf.fit(X_train_balanced, y_train_balanced)

stacking_clf = StackingClassifier(
    estimators=[
        ('xgb', trained_models['XGBoost']),
        ('lgb', trained_models['LightGBM']),
        ('cb', trained_models['CatBoost']),
        ('rf', trained_models['Random Forest'])
    ],
    final_estimator=LogisticRegression(),
    cv=5
)

stacking_clf.fit(X_train_balanced, y_train_balanced)

# Evaluate ensembles
ensemble_models = {
    'Voting Classifier': voting_clf,
    'Stacking Classifier': stacking_clf
}

best_model = None
best_score = 0
best_name = ""

print("\nEvaluating ensemble models...")
for name, model in ensemble_models.items():
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_name = name

print(f"\n{'='*50}")
print(f"BEST MODEL: {best_name}")
print(f"{'='*50}")

# Final evaluation
y_pred_best = best_model.predict(X_test_scaled)
y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:\n", classification_report(y_test, y_pred_best))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best))

# Save
print("\nSaving model...")
joblib.dump(best_model, "best_diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(selector, "feature_selector.pkl")
with open("selected_features.txt", "w") as f:
    f.write("\n".join(selected_features))
print("All components saved successfully.")

import joblib

model = joblib.load("best_diabetes_model.pkl")
print(model)