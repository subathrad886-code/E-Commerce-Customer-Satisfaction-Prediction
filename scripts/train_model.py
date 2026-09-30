"""
=============================================================================
SUBATHRA D - E-Commerce Customer Satisfaction Prediction
Machine Learning Training, Evaluation & Model Selection Script
=============================================================================
This script:
1. Loads and preprocesses the customer satisfaction dataset
2. Builds an end-to-end ColumnTransformer preprocessing pipeline
3. Implements & trains two classification algorithms:
     - Logistic Regression
     - Random Forest Classifier
4. Evaluates both models across:
     - Accuracy
     - Precision (Weighted & Macro)
     - Recall (Weighted & Macro)
     - F1-Score (Weighted & Macro)
     - Confusion Matrices
5. Generates comparison visualizations saved to static/eda/
6. Automatically selects the highest-performing model
7. Saves the best model pipeline and preprocessor to models/
=============================================================================
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_path = os.path.join(project_root, "data", "customer_satisfaction.csv")
    models_dir = os.path.join(project_root, "models")
    eda_dir = os.path.join(project_root, "static", "eda")
    
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(eda_dir, exist_ok=True)
    
    print("=" * 80)
    print("SUBATHRA D - E-Commerce Customer Satisfaction: ML Model Training & Comparison")
    print("=" * 80)
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset file not found at: {data_path}")
        print("Please run first: python scripts/generate_dataset.py")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    print(f"[*] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # -----------------------------------------------------------------
    # 1. Feature & Target Separation
    # -----------------------------------------------------------------
    target_col = "customer_satisfaction"
    drop_cols = ["customer_id", target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    classes = ["Low", "Medium", "High"]
    
    numerical_cols = [
        "customer_age",
        "purchase_amount",
        "delivery_time",
        "service_interaction",
        "previous_orders",
        "website_rating"
    ]
    
    categorical_cols = [
        "gender",
        "product_category",
        "discount_used",
        "payment_method",
        "return_requested",
        "order_frequency"
    ]
    
    print(f"[*] Feature Configuration:")
    print(f"    - Numerical Features ({len(numerical_cols)}):   {', '.join(numerical_cols)}")
    print(f"    - Categorical Features ({len(categorical_cols)}): {', '.join(categorical_cols)}")
    print(f"    - Target Classes ({len(classes)}):        {', '.join(classes)}")
    
    # -----------------------------------------------------------------
    # 2. Data Preprocessing Pipelines
    # -----------------------------------------------------------------
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipeline, numerical_cols),
        ("cat", cat_pipeline, categorical_cols)
    ])
    
    # -----------------------------------------------------------------
    # 3. Train-Test Split (Stratified 80/20)
    # -----------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"\n[*] Train-Test Partition (80/20 Stratified Split):")
    print(f"    - Training Samples: {len(X_train)}")
    print(f"    - Testing Samples:  {len(X_test)}")
    
    # -----------------------------------------------------------------
    # 4. Model Definitions
    # -----------------------------------------------------------------
    classifiers = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            random_state=42
        )
    }
    
    metrics_summary = {}
    trained_pipelines = {}
    confusion_matrices = {}
    
    print("\n" + "=" * 80)
    print("TRAINING AND EVALUATING MODELS")
    print("=" * 80)
    
    for name, clf in classifiers.items():
        print(f"\n[>>>] Training: {name}...")
        # Full pipeline combining preprocessor + classifier
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        
        # Fit on training data
        pipeline.fit(X_train, y_train)
        
        # Predict on test data
        y_pred = pipeline.predict(X_test)
        
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred) * 100
        prec_w = precision_score(y_test, y_pred, average="weighted", zero_division=0) * 100
        rec_w = recall_score(y_test, y_pred, average="weighted", zero_division=0) * 100
        f1_w = f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100
        
        prec_m = precision_score(y_test, y_pred, average="macro", zero_division=0) * 100
        rec_m = recall_score(y_test, y_pred, average="macro", zero_division=0) * 100
        f1_m = f1_score(y_test, y_pred, average="macro", zero_division=0) * 100
        
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        confusion_matrices[name] = cm
        
        metrics_summary[name] = {
            "accuracy": round(acc, 2),
            "precision_weighted": round(prec_w, 2),
            "recall_weighted": round(rec_w, 2),
            "f1_weighted": round(f1_w, 2),
            "precision_macro": round(prec_m, 2),
            "recall_macro": round(rec_m, 2),
            "f1_macro": round(f1_m, 2),
            "confusion_matrix": cm.tolist()
        }
        trained_pipelines[name] = pipeline
        
        print(f"    - Accuracy:           {acc:.2f}%")
        print(f"    - Precision (Weighted): {prec_w:.2f}%")
        print(f"    - Recall (Weighted):    {rec_w:.2f}%")
        print(f"    - F1-Score (Weighted):  {f1_w:.2f}%")
        print(f"    - Precision (Macro):    {prec_m:.2f}%")
        print(f"    - Recall (Macro):       {rec_m:.2f}%")
        print(f"    - F1-Score (Macro):     {f1_m:.2f}%")
        
        print(f"\n    Detailed Classification Report ({name}):")
        report = classification_report(y_test, y_pred, labels=classes, zero_division=0)
        for line in report.split("\n"):
            print(f"      {line}")
            
    # -----------------------------------------------------------------
    # 5. Model Comparison Table
    # -----------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MODEL COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Algorithm':<22} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 74)
    for name, m in metrics_summary.items():
        print(f"{name:<22} | {m['accuracy']:>8.2f}% | {m['precision_weighted']:>8.2f}% | {m['recall_weighted']:>8.2f}% | {m['f1_weighted']:>8.2f}%")
    print("-" * 74)
    
    # -----------------------------------------------------------------
    # 6. Automatic Model Selection
    # -----------------------------------------------------------------
    # Select best model primarily based on weighted F1-Score, tie-break on accuracy
    best_model_name = max(
        metrics_summary.keys(),
        key=lambda k: (metrics_summary[k]["f1_weighted"], metrics_summary[k]["accuracy"])
    )
    best_pipeline = trained_pipelines[best_model_name]
    
    print(f"\n[WINNER] Selected Best Model: >>> {best_model_name} <<<")
    print(f"         F1-Score: {metrics_summary[best_model_name]['f1_weighted']:.2f}% | Accuracy: {metrics_summary[best_model_name]['accuracy']:.2f}%")
    
    # -----------------------------------------------------------------
    # 7. Save Visualizations (Confusion Matrices & Comparison Chart)
    # -----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for idx, (name, cm) in enumerate(confusion_matrices.items()):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues" if idx == 0 else "Greens",
                    xticklabels=classes, yticklabels=classes, ax=ax,
                    cbar=False, annot_kws={"size": 13, "weight": "bold"})
        ax.set_title(f"{name}\nConfusion Matrix", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Predicted Satisfaction", fontweight="semibold")
        ax.set_ylabel("Actual Satisfaction", fontweight="semibold")
    plt.tight_layout()
    cm_path = os.path.join(eda_dir, "confusion_matrix_comparison.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"\n[OK] Saved confusion matrix comparison plot to: {cm_path}")
    
    # Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(9, 5))
    metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(metrics_names))
    width = 0.35
    
    lr_vals = [
        metrics_summary["Logistic Regression"]["accuracy"],
        metrics_summary["Logistic Regression"]["precision_weighted"],
        metrics_summary["Logistic Regression"]["recall_weighted"],
        metrics_summary["Logistic Regression"]["f1_weighted"]
    ]
    rf_vals = [
        metrics_summary["Random Forest"]["accuracy"],
        metrics_summary["Random Forest"]["precision_weighted"],
        metrics_summary["Random Forest"]["recall_weighted"],
        metrics_summary["Random Forest"]["f1_weighted"]
    ]
    
    rects1 = ax.bar(x - width/2, lr_vals, width, label="Logistic Regression", color="#3b82f6", edgecolor="#1e3a8a")
    rects2 = ax.bar(x + width/2, rf_vals, width, label="Random Forest", color="#10b981", edgecolor="#065f46")
    
    ax.set_ylabel("Score (%)", fontweight="semibold")
    ax.set_title("Model Performance Comparison (Test Set)", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontweight="semibold")
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    
    for bar in list(rects1) + list(rects2):
        height = bar.get_height()
        ax.annotate(f"{height:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
                    
    plt.tight_layout()
    comp_path = os.path.join(eda_dir, "model_comparison_bar.png")
    fig.savefig(comp_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved model comparison bar plot to: {comp_path}")
    
    # -----------------------------------------------------------------
    # 8. Save Model Artifacts
    # -----------------------------------------------------------------
    # Save the full best pipeline (includes preprocessor + model)
    model_pkl_path = os.path.join(models_dir, "customer_satisfaction_model.pkl")
    joblib.dump(best_pipeline, model_pkl_path)
    print(f"[OK] Saved Best Pipeline to: {model_pkl_path}")
    
    # Save preprocessor separately as requested
    preprocessor_fitted = best_pipeline.named_steps["preprocessor"]
    preprocessor_pkl_path = os.path.join(models_dir, "preprocessor.pkl")
    joblib.dump(preprocessor_fitted, preprocessor_pkl_path)
    print(f"[OK] Saved Preprocessor to:   {preprocessor_pkl_path}")
    
    # Save metrics JSON
    metrics_json_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "metrics": metrics_summary,
            "classes": classes,
            "numerical_cols": numerical_cols,
            "categorical_cols": categorical_cols,
            "feature_order": feature_cols
        }, f, indent=4)
    print(f"[OK] Saved Evaluation Metrics to: {metrics_json_path}")
    
    print("\n" + "=" * 80)
    print("[SUCCESS] Machine learning training and evaluation completed successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
