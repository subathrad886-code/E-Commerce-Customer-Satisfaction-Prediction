"""
=============================================================================
SUBATHRA D - E-Commerce Customer Satisfaction Prediction
Exploratory Data Analysis (EDA) & Visualization Script
=============================================================================
This script performs in-depth Exploratory Data Analysis on the customer
satisfaction dataset and saves 8 publication-ready visual charts into:
    static/eda/
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for scripts & servers
import matplotlib.pyplot as plt
import seaborn as sns

# Color palette mapped to satisfaction categories
SATISFACTION_PALETTE = {
    "High": "#10b981",    # Emerald Green
    "Medium": "#f59e0b",  # Amber Yellow
    "Low": "#ef4444"      # Coral Red
}

def setup_plot_style():
    """Configures clean, modern plot aesthetics."""
    sns.set_theme(style="whitegrid", font="sans-serif")
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelweight": "semibold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.autolayout": False
    })

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_path = os.path.join(project_root, "data", "customer_satisfaction.csv")
    output_dir = os.path.join(project_root, "static", "eda")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 75)
    print("SUBATHRA D - E-Commerce Customer Satisfaction: Exploratory Data Analysis")
    print("=" * 75)
    
    if not os.path.exists(data_path):
        print(f"[ERROR] Dataset not found at: {data_path}")
        print("Please run first: python scripts/generate_dataset.py")
        sys.exit(1)
        
    df = pd.read_csv(data_path)
    setup_plot_style()
    
    # -------------------------------------------------------------
    # 1. Dataset Information
    # -------------------------------------------------------------
    print("\n[1] DATASET OVERVIEW & SHAPE:")
    print(f"    - Total Rows: {df.shape[0]}")
    print(f"    - Total Columns: {df.shape[1]}")
    print("\n[2] DATA TYPES:")
    for col, dtype in df.dtypes.items():
        print(f"    - {col:25s}: {str(dtype)}")
        
    # -------------------------------------------------------------
    # 2. Missing Value Analysis
    # -------------------------------------------------------------
    print("\n[3] MISSING VALUE ANALYSIS:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("    [OK] No missing values detected across all columns.")
    else:
        for col, cnt in missing[missing > 0].items():
            print(f"    - {col:25s}: {cnt} missing")
            
    # -------------------------------------------------------------
    # 3. Descriptive Statistics
    # -------------------------------------------------------------
    print("\n[4] NUMERICAL FEATURES SUMMARY STATISTICS:")
    print(df.describe().round(2).to_string())
    
    # -------------------------------------------------------------
    # 4. Satisfaction Category Distribution
    # -------------------------------------------------------------
    print("\n[5] TARGET DISTRIBUTION (customer_satisfaction):")
    counts = df["customer_satisfaction"].value_counts()
    for cat, count in counts.items():
        pct = (count / len(df)) * 100
        print(f"    - {cat:8s}: {count:4d} ({pct:.1f}%)")
        
    print(f"\n[*] Generating and saving EDA visualizations to '{output_dir}'...")
    
    # -------------------------------------------------------------
    # Chart 1: Satisfaction Category Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["Low", "Medium", "High"]
    cat_counts = df["customer_satisfaction"].value_counts().reindex(order)
    colors = [SATISFACTION_PALETTE[c] for c in order]
    bars = ax.bar(order, cat_counts, color=colors, edgecolor="#333333", linewidth=1.2, width=0.55)
    for bar in bars:
        height = bar.get_height()
        pct = (height / len(df)) * 100
        ax.annotate(f"{height}\n({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height / 2),
                    xytext=(0, 0), textcoords="offset points",
                    ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    ax.set_title("Customer Satisfaction Category Distribution", pad=15)
    ax.set_xlabel("Satisfaction Level")
    ax.set_ylabel("Number of Customers")
    ax.set_ylim(0, cat_counts.max() * 1.2)
    plt.tight_layout()
    chart1_path = os.path.join(output_dir, "satisfaction_distribution.png")
    fig.savefig(chart1_path, dpi=300)
    plt.close(fig)
    print(f"    [OK] Saved: satisfaction_distribution.png")
    
    # -------------------------------------------------------------
    # Chart 2: Age Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["customer_age"], bins=20, kde=True, color="#3b82f6", ax=ax, edgecolor="#1e3a8a")
    mean_age = df["customer_age"].mean()
    median_age = df["customer_age"].median()
    ax.axvline(mean_age, color="#ef4444", linestyle="--", linewidth=2, label=f"Mean: {mean_age:.1f}")
    ax.axvline(median_age, color="#10b981", linestyle="-.", linewidth=2, label=f"Median: {median_age:.1f}")
    ax.set_title("Customer Age Distribution", pad=15)
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Customer Count")
    ax.legend()
    plt.tight_layout()
    chart2_path = os.path.join(output_dir, "age_distribution.png")
    fig.savefig(chart2_path, dpi=300)
    plt.close(fig)
    print(f"    [OK] Saved: age_distribution.png")
    
    # -------------------------------------------------------------
    # Chart 3: Purchase Amount Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df["purchase_amount"], bins=25, kde=True, color="#8b5cf6", ax=ax, edgecolor="#4c1d95")
    mean_amt = df["purchase_amount"].mean()
    median_amt = df["purchase_amount"].median()
    ax.axvline(mean_amt, color="#ef4444", linestyle="--", linewidth=2, label=f"Mean: Rs. {mean_amt:,.0f}")
    ax.axvline(median_amt, color="#10b981", linestyle="-.", linewidth=2, label=f"Median: Rs. {median_amt:,.0f}")
    ax.set_title("Customer Purchase Amount Distribution", pad=15)
    ax.set_xlabel("Purchase Amount (Rs.)")
    ax.set_ylabel("Customer Count")
    ax.legend()
    plt.tight_layout()
    chart3_path = os.path.join(output_dir, "purchase_amount_distribution.png")
    fig.savefig(chart3_path, dpi=300)
    plt.close(fig)
    print(f"    [OK] Saved: purchase_amount_distribution.png")
    
    # -------------------------------------------------------------
    # Chart 4: Delivery Time Distribution
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    delivery_counts = df["delivery_time"].value_counts().sort_index()
    ax.bar(delivery_counts.index, delivery_counts.values, color="#06b6d4", edgecolor="#0e7490", width=0.65)
    ax.set_title("Delivery Time Distribution", pad=15)
    ax.set_xlabel("Delivery Time (Days)")
    ax.set_ylabel("Number of Orders")
    ax.set_xticks(delivery_counts.index)
    for x, y in zip(delivery_counts.index, delivery_counts.values):
        ax.text(x, y + 2, str(y), ha="center", va="bottom", fontsize=9, fontweight="semibold")
    ax.set_ylim(0, delivery_counts.max() * 1.15)
    plt.tight_layout()
    chart4_path = os.path.join(output_dir, "delivery_time_distribution.png")
    fig.savefig(chart4_path, dpi=300)
    plt.close(fig)
    print(f"    [OK] Saved: delivery_time_distribution.png")
    
    # -------------------------------------------------------------
    # Chart 5: Satisfaction vs Product Category
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5.5))
    cat_sat = pd.crosstab(df["product_category"], df["customer_satisfaction"], normalize="index") * 100
    cat_sat = cat_sat[["Low", "Medium", "High"]]  # specific order
    cat_sat.plot(kind="bar", stacked=True, color=[SATISFACTION_PALETTE[c] for c in ["Low", "Medium", "High"]],
                 ax=ax, edgecolor="#333333", width=0.65)
    ax.set_title("Satisfaction Distribution by Product Category (%)", pad=15)
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend(title="Satisfaction Level", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    chart5_path = os.path.join(output_dir, "satisfaction_vs_category.png")
    fig.savefig(chart5_path, dpi=300)
    plt.close(fig)
    print(f"    [OK] Saved: satisfaction_vs_category.png")
    
    # -------------------------------------------------------------
    # Chart 6: Satisfaction vs Delivery Time
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    order = ["Low", "Medium", "High"]
    sns.boxplot(data=df, x="customer_satisfaction", y="delivery_time", order=order,
                hue="customer_satisfaction", palette=SATISFACTION_PALETTE, ax=ax, width=0.5,
                boxprops=dict(alpha=0.9), legend=False)
    sns.stripplot(data=df, x="customer_satisfaction", y="delivery_time", order=order,
                  color="black", alpha=0.25, jitter=0.2, size=4, ax=ax)
    ax.set_title("Delivery Time across Satisfaction Levels", pad=15)
    ax.set_xlabel("Customer Satisfaction")
    ax.set_ylabel("Delivery Time (Days)")
    plt.tight_layout()
    chart6_path = os.path.join(output_dir, "satisfaction_vs_delivery.png")
    fig.savefig(chart6_path, dpi=300)
    plt.close(fig)
    print(f"    [OK] Saved: satisfaction_vs_delivery.png")
    
    # -------------------------------------------------------------
    # Chart 7: Satisfaction vs Purchase Amount
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5))
    order = ["Low", "Medium", "High"]
    sns.violinplot(data=df, x="customer_satisfaction", y="purchase_amount", order=order,
                   hue="customer_satisfaction", palette=SATISFACTION_PALETTE, ax=ax,
                   inner="quartile", cut=0, legend=False)
    ax.set_title("Purchase Amount across Satisfaction Levels", pad=15)
    ax.set_xlabel("Customer Satisfaction")
    ax.set_ylabel("Purchase Amount (Rs.)")
    plt.tight_layout()
    chart7_path = os.path.join(output_dir, "satisfaction_vs_purchase.png")
    fig.savefig(chart7_path, dpi=300)
    plt.close(fig)
    print(f"    [OK] Saved: satisfaction_vs_purchase.png")
    
    # -------------------------------------------------------------
    # Chart 8: Satisfaction vs Service Interaction
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8.5, 5))
    serv_sat = pd.crosstab(df["service_interaction"], df["customer_satisfaction"], normalize="index") * 100
    serv_sat = serv_sat[["Low", "Medium", "High"]]
    serv_sat.plot(kind="bar", stacked=True, color=[SATISFACTION_PALETTE[c] for c in ["Low", "Medium", "High"]],
                  ax=ax, edgecolor="#333333", width=0.6)
    ax.set_title("Customer Satisfaction vs Service Interaction Rating", pad=15)
    ax.set_xlabel("Service Interaction Rating (1 = Very Poor, 5 = Excellent)")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticklabels(["1 (Poor)", "2 (Below Avg)", "3 (Average)", "4 (Good)", "5 (Excellent)"], rotation=0)
    ax.legend(title="Satisfaction", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    chart8_path = os.path.join(output_dir, "satisfaction_vs_service.png")
    fig.savefig(chart8_path, dpi=300)
    plt.close(fig)
    print(f"    [OK] Saved: satisfaction_vs_service.png")
    
    print("\n" + "=" * 75)
    print("[SUCCESS] All 8 EDA visual charts successfully generated & saved to static/eda/!")
    print("=" * 75)

if __name__ == "__main__":
    main()
