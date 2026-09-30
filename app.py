"""
=============================================================================
SUBATHRA D - E-Commerce Customer Satisfaction Prediction
Flask Backend Web Application
=============================================================================
This application integrates:
- Scikit-learn Machine Learning Model Pipeline
- SQLite Database for prediction audit storage
- Interactive Prediction Frontend
- Real-time Analytics Dashboard with Chart.js
- REST APIs for predictions and dashboard metrics
=============================================================================
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for

# -----------------------------------------------------------------------------
# Configuration and Relative Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "customer_satisfaction.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "customer_satisfaction_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")
DB_DIR = os.path.join(BASE_DIR, "database")
DB_PATH = os.path.join(DB_DIR, "customer_satisfaction.db")

app = Flask(__name__)
app.config["SECRET_KEY"] = "subathra-ecommerce-satisfaction-secret-key-2026"

# -----------------------------------------------------------------------------
# Lazy Model & Metrics Loader with Friendly Error Messages
# -----------------------------------------------------------------------------
model_pipeline = None
model_metadata = {}

def get_model():
    """Safely loads or retrieves the trained ML model pipeline."""
    global model_pipeline, model_metadata
    if model_pipeline is not None:
        return model_pipeline, None
        
    if not os.path.exists(MODEL_PATH):
        err = "Trained model not found at 'models/customer_satisfaction_model.pkl'. Please run: python scripts/train_model.py"
        return None, err
        
    try:
        model_pipeline = joblib.load(MODEL_PATH)
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH, "r") as f:
                model_metadata = json.load(f)
        return model_pipeline, None
    except Exception as e:
        return None, f"Failed to load machine learning model: {str(e)}"

# -----------------------------------------------------------------------------
# SQLite Database Setup & Helper Functions
# -----------------------------------------------------------------------------
def get_db_connection():
    """Establishes and returns an SQLite connection with Row factory."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schema and seeds initial records if empty."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_age INTEGER NOT NULL,
            gender TEXT NOT NULL,
            product_category TEXT NOT NULL,
            purchase_amount REAL NOT NULL,
            delivery_time INTEGER NOT NULL,
            service_interaction INTEGER NOT NULL,
            previous_orders INTEGER NOT NULL,
            discount_used TEXT NOT NULL,
            payment_method TEXT NOT NULL,
            website_rating INTEGER NOT NULL,
            return_requested TEXT NOT NULL,
            order_frequency TEXT NOT NULL,
            predicted_satisfaction TEXT NOT NULL,
            prediction_probability REAL NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    
    # Check if empty, seed with starter demo records so dashboard is instantly rich
    count = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    if count == 0 and os.path.exists(DATA_PATH):
        try:
            df_seed = pd.read_csv(DATA_PATH).head(15)
            # Use model to predict if available, or use the target column
            model, _ = get_model()
            
            for _, row in df_seed.iterrows():
                pred = row.get("customer_satisfaction", "Medium")
                prob = 85.0
                if model:
                    row_df = pd.DataFrame([row.drop(labels=["customer_id", "customer_satisfaction"], errors="ignore")])
                    pred = model.predict(row_df)[0]
                    if hasattr(model, "predict_proba"):
                        prob = round(float(np.max(model.predict_proba(row_df)[0])) * 100, 2)
                        
                cursor.execute("""
                    INSERT INTO predictions (
                        customer_age, gender, product_category, purchase_amount,
                        delivery_time, service_interaction, previous_orders, discount_used,
                        payment_method, website_rating, return_requested, order_frequency,
                        predicted_satisfaction, prediction_probability, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    int(row["customer_age"]), str(row["gender"]), str(row["product_category"]),
                    float(row["purchase_amount"]), int(row["delivery_time"]), int(row["service_interaction"]),
                    int(row["previous_orders"]), str(row["discount_used"]), str(row["payment_method"]),
                    int(row["website_rating"]), str(row["return_requested"]), str(row["order_frequency"]),
                    pred, prob, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ))
            conn.commit()
            print("[INFO] Seeded database with 15 initial customer predictions.")
        except Exception as e:
            print(f"[WARNING] Could not auto-seed database: {e}")
            
    conn.close()

# Initialize DB on import/startup
init_db()

# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route("/")
def home():
    """Home prediction page with full interactive form and status banner."""
    model, err = get_model()
    
    model_name = "Logistic Regression"
    model_accuracy = 76.0
    if model_metadata and "best_model" in model_metadata:
        model_name = model_metadata["best_model"]
        if "metrics" in model_metadata and model_name in model_metadata["metrics"]:
            model_accuracy = model_metadata["metrics"][model_name]["accuracy"]
            
    return render_template(
        "index.html",
        model_name=model_name,
        model_accuracy=model_accuracy,
        error_msg=err
    )

@app.route("/predict", methods=["POST"])
def predict():
    """Processes customer information submitted via the frontend form."""
    model, err = get_model()
    if err:
        return render_template("index.html", error_msg=err)
        
    try:
        form = request.form
        customer_data = {
            "customer_age": int(form.get("customer_age", 30)),
            "gender": str(form.get("gender", "Male")),
            "product_category": str(form.get("product_category", "Electronics")),
            "purchase_amount": float(form.get("purchase_amount", 1000.0)),
            "delivery_time": int(form.get("delivery_time", 3)),
            "service_interaction": int(form.get("service_interaction", 3)),
            "previous_orders": int(form.get("previous_orders", 2)),
            "discount_used": str(form.get("discount_used", "No")),
            "payment_method": str(form.get("payment_method", "UPI")),
            "website_rating": int(form.get("website_rating", 3)),
            "return_requested": str(form.get("return_requested", "No")),
            "order_frequency": str(form.get("order_frequency", "Monthly"))
        }
        
        # Build DataFrame matching training features
        df_input = pd.DataFrame([customer_data])
        
        # Predict class
        predicted_class = model.predict(df_input)[0]
        
        # Predict probability
        confidence = 80.0
        prob_dict = {}
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df_input)[0]
            confidence = round(float(np.max(probs)) * 100, 2)
            classes = model.classes_
            for cls_name, p in zip(classes, probs):
                prob_dict[cls_name] = round(float(p) * 100, 1)
        else:
            prob_dict = {predicted_class: 100.0}
            
        # Store prediction in SQLite
        conn = get_db_connection()
        cursor = conn.cursor()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        cursor.execute("""
            INSERT INTO predictions (
                customer_age, gender, product_category, purchase_amount,
                delivery_time, service_interaction, previous_orders, discount_used,
                payment_method, website_rating, return_requested, order_frequency,
                predicted_satisfaction, prediction_probability, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            customer_data["customer_age"], customer_data["gender"], customer_data["product_category"],
            customer_data["purchase_amount"], customer_data["delivery_time"], customer_data["service_interaction"],
            customer_data["previous_orders"], customer_data["discount_used"], customer_data["payment_method"],
            customer_data["website_rating"], customer_data["return_requested"], customer_data["order_frequency"],
            predicted_class, confidence, created_at
        ))
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        
        return render_template(
            "result.html",
            prediction=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            customer=customer_data,
            record_id=record_id
        )
        
    except Exception as e:
        return render_template("index.html", error_msg=f"Prediction error: {str(e)}")

@app.route("/dashboard")
def dashboard():
    """Displays comprehensive analytics synthesized from database records and model metrics."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Total count
    total_predictions = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    
    if total_predictions == 0:
        high_count = medium_count = low_count = 0
        high_pct = medium_pct = low_pct = 0.0
        avg_purchase = avg_delivery = avg_website_rating = 0.0
    else:
        # Category counts
        cat_counts = dict(cursor.execute("""
            SELECT predicted_satisfaction, COUNT(*) 
            FROM predictions 
            GROUP BY predicted_satisfaction
        """).fetchall())
        
        high_count = cat_counts.get("High", 0)
        medium_count = cat_counts.get("Medium", 0)
        low_count = cat_counts.get("Low", 0)
        
        high_pct = round((high_count / total_predictions) * 100, 1)
        medium_pct = round((medium_count / total_predictions) * 100, 1)
        low_pct = round((low_count / total_predictions) * 100, 1)
        
        # Averages
        avg_row = cursor.execute("""
            SELECT AVG(purchase_amount), AVG(delivery_time), AVG(website_rating) 
            FROM predictions
        """).fetchone()
        
        avg_purchase = round(float(avg_row[0] or 0), 2)
        avg_delivery = round(float(avg_row[1] or 0), 1)
        avg_website_rating = round(float(avg_row[2] or 0), 2)
        
    # 2. Charts Data Preparation
    # A. Satisfaction Distribution
    distribution_data = {
        "High": high_count,
        "Medium": medium_count,
        "Low": low_count
    }
    
    # B. Satisfaction by Product Category
    categories = [
        "Electronics", "Fashion", "Home & Kitchen",
        "Beauty & Personal Care", "Books & Stationery", "Sports & Fitness"
    ]
    by_category_data = {
        "categories": categories,
        "high": [],
        "medium": [],
        "low": []
    }
    
    for cat in categories:
        c_high = cursor.execute("SELECT COUNT(*) FROM predictions WHERE product_category = ? AND predicted_satisfaction = 'High'", (cat,)).fetchone()[0]
        c_med = cursor.execute("SELECT COUNT(*) FROM predictions WHERE product_category = ? AND predicted_satisfaction = 'Medium'", (cat,)).fetchone()[0]
        c_low = cursor.execute("SELECT COUNT(*) FROM predictions WHERE product_category = ? AND predicted_satisfaction = 'Low'", (cat,)).fetchone()[0]
        by_category_data["high"].append(c_high)
        by_category_data["medium"].append(c_med)
        by_category_data["low"].append(c_low)
        
    # C. Avg Delivery Time by Satisfaction
    avg_delivery_by_sat = {}
    for sat in ["High", "Medium", "Low"]:
        row = cursor.execute("SELECT AVG(delivery_time) FROM predictions WHERE predicted_satisfaction = ?", (sat,)).fetchone()
        avg_delivery_by_sat[sat] = round(float(row[0] or 0), 1)
        
    # D. Avg Purchase Amount by Satisfaction
    avg_purchase_by_sat = {}
    for sat in ["High", "Medium", "Low"]:
        row = cursor.execute("SELECT AVG(purchase_amount) FROM predictions WHERE predicted_satisfaction = ?", (sat,)).fetchone()
        avg_purchase_by_sat[sat] = round(float(row[0] or 0), 2)
        
    # E. Satisfaction by Gender
    by_gender = {}
    for g in ["Female", "Male", "Other"]:
        by_gender[g] = {
            "High": cursor.execute("SELECT COUNT(*) FROM predictions WHERE gender = ? AND predicted_satisfaction = 'High'", (g,)).fetchone()[0],
            "Medium": cursor.execute("SELECT COUNT(*) FROM predictions WHERE gender = ? AND predicted_satisfaction = 'Medium'", (g,)).fetchone()[0],
            "Low": cursor.execute("SELECT COUNT(*) FROM predictions WHERE gender = ? AND predicted_satisfaction = 'Low'", (g,)).fetchone()[0]
        }
        
    conn.close()
    
    chart_payload = {
        "distribution": distribution_data,
        "by_category": by_category_data,
        "avg_delivery_by_sat": avg_delivery_by_sat,
        "avg_purchase_by_sat": avg_purchase_by_sat,
        "by_gender": by_gender
    }
    
    # Model evaluation metrics from metadata
    _, _ = get_model()
    model_metrics = model_metadata.get("metrics", {})
    best_model_name = model_metadata.get("best_model", "Logistic Regression")
    
    return render_template(
        "dashboard.html",
        total_predictions=total_predictions,
        high_count=high_count,
        medium_count=medium_count,
        low_count=low_count,
        high_pct=high_pct,
        medium_pct=medium_pct,
        low_pct=low_pct,
        avg_purchase=avg_purchase,
        avg_delivery=avg_delivery,
        avg_website_rating=avg_website_rating,
        chart_payload=chart_payload,
        model_metrics=model_metrics,
        best_model_name=best_model_name
    )

@app.route("/history")
def history():
    """Displays audit history of all prediction records from SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    records = cursor.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    conn.close()
    return render_template("history.html", records=records)

@app.route("/api/stats")
def api_stats():
    """REST API endpoint returning summary database statistics."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    total = cursor.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    cat_counts = dict(cursor.execute("""
        SELECT predicted_satisfaction, COUNT(*) 
        FROM predictions 
        GROUP BY predicted_satisfaction
    """).fetchall())
    
    avg_row = cursor.execute("""
        SELECT AVG(purchase_amount), AVG(delivery_time), AVG(website_rating) 
        FROM predictions
    """).fetchone()
    
    conn.close()
    
    return jsonify({
        "status": "success",
        "total_predictions": total,
        "satisfaction_breakdown": {
            "High": cat_counts.get("High", 0),
            "Medium": cat_counts.get("Medium", 0),
            "Low": cat_counts.get("Low", 0)
        },
        "averages": {
            "purchase_amount": round(float(avg_row[0] or 0), 2),
            "delivery_time": round(float(avg_row[1] or 0), 1),
            "website_rating": round(float(avg_row[2] or 0), 2)
        }
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """REST API endpoint accepting JSON payloads for programmatic prediction."""
    model, err = get_model()
    if err:
        return jsonify({"status": "error", "message": err}), 500
        
    try:
        data = request.get_json(force=True)
        required_fields = [
            "customer_age", "gender", "product_category", "purchase_amount",
            "delivery_time", "service_interaction", "previous_orders",
            "discount_used", "payment_method", "website_rating",
            "return_requested", "order_frequency"
        ]
        
        for rf in required_fields:
            if rf not in data:
                return jsonify({"status": "error", "message": f"Missing required field: {rf}"}), 400
                
        df_input = pd.DataFrame([data])
        prediction = model.predict(df_input)[0]
        
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            confidence = round(float(np.max(model.predict_proba(df_input)[0])) * 100, 2)
            
        # Store in DB
        conn = get_db_connection()
        cursor = conn.cursor()
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("""
            INSERT INTO predictions (
                customer_age, gender, product_category, purchase_amount,
                delivery_time, service_interaction, previous_orders, discount_used,
                payment_method, website_rating, return_requested, order_frequency,
                predicted_satisfaction, prediction_probability, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(data["customer_age"]), str(data["gender"]), str(data["product_category"]),
            float(data["purchase_amount"]), int(data["delivery_time"]), int(data["service_interaction"]),
            int(data["previous_orders"]), str(data["discount_used"]), str(data["payment_method"]),
            int(data["website_rating"]), str(data["return_requested"]), str(data["order_frequency"]),
            prediction, confidence, created_at
        ))
        conn.commit()
        record_id = cursor.lastrowid
        conn.close()
        
        return jsonify({
            "status": "success",
            "record_id": record_id,
            "predicted_satisfaction": prediction,
            "prediction_probability": confidence,
            "created_at": created_at
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

# -----------------------------------------------------------------------------
# Main Application Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("SUBATHRA D - E-Commerce Customer Satisfaction Prediction Server")
    print("=" * 70)
    print("[*] Starting Flask Development Server...")
    print("[*] Access Application at: http://127.0.0.1:5000")
    print("[*] Analytics Dashboard:   http://127.0.0.1:5000/dashboard")
    print("[*] Prediction History:    http://127.0.0.1:5000/history")
    print("=" * 70)
    app.run(host="127.0.0.1", port=5000, debug=True)
