"""
=============================================================================
SUBATHRA D – E-Commerce Customer Satisfaction Prediction
Dataset Generation Script
=============================================================================
This script generates a realistic synthetic dataset of 1,000 e-commerce
customer records with realistic behavioral, transactional, and satisfaction
patterns.

Target Column: customer_satisfaction (Low, Medium, High)
Output File:   data/customer_satisfaction.csv
=============================================================================
"""

import os
import numpy as np
import pandas as pd

def generate_customer_satisfaction_data(n_samples=1000, random_seed=42):
    """
    Generates a realistic synthetic dataset for customer satisfaction prediction.
    
    Parameters:
        n_samples (int): Number of customer records to generate (default: 1000).
        random_seed (int): Seed for reproducibility.
        
    Returns:
        pd.DataFrame: Generated dataset.
    """
    np.random.seed(random_seed)
    
    # 1. Customer Identification
    customer_ids = [f"CUST-{1001 + i}" for i in range(n_samples)]
    
    # 2. Demographics
    ages = np.random.randint(18, 71, size=n_samples)
    genders = np.random.choice(["Female", "Male", "Other"], size=n_samples, p=[0.49, 0.47, 0.04])
    
    # 3. Product & Shopping Profile
    categories = [
        "Electronics",
        "Fashion",
        "Home & Kitchen",
        "Beauty & Personal Care",
        "Books & Stationery",
        "Sports & Fitness"
    ]
    product_categories = np.random.choice(categories, size=n_samples, p=[0.25, 0.25, 0.18, 0.14, 0.10, 0.08])
    
    # Purchase Amount (INR / USD equivalent scaled nicely)
    # Higher for Electronics and Home, moderate for Fashion/Sports, lower for Books
    purchase_amounts = []
    for cat in product_categories:
        if cat == "Electronics":
            amt = np.random.uniform(2500, 48000)
        elif cat == "Home & Kitchen":
            amt = np.random.uniform(1200, 28000)
        elif cat == "Fashion":
            amt = np.random.uniform(800, 15000)
        elif cat == "Sports & Fitness":
            amt = np.random.uniform(600, 18000)
        elif cat == "Beauty & Personal Care":
            amt = np.random.uniform(400, 9500)
        else:  # Books & Stationery
            amt = np.random.uniform(250, 4500)
        purchase_amounts.append(round(amt, 2))
    purchase_amounts = np.array(purchase_amounts)
    
    # 4. Delivery & Logistics
    # Days taken for delivery: 1 to 14 days
    delivery_times = np.random.choice(
        np.arange(1, 15),
        size=n_samples,
        p=[0.08, 0.14, 0.18, 0.17, 0.14, 0.10, 0.07, 0.04, 0.03, 0.02, 0.01, 0.01, 0.005, 0.005]
    )
    
    # 5. Service & Experience Ratings
    # Customer service interaction rating: 1 (Very Poor) to 5 (Excellent)
    service_interactions = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.12, 0.18, 0.30, 0.24, 0.16])
    
    # Previous order history
    previous_orders = np.random.choice(
        np.arange(0, 26),
        size=n_samples,
        p=[0.15] + [0.85 / 25] * 25
    )
    
    # Discounts
    discounts_used = np.random.choice(["Yes", "No"], size=n_samples, p=[0.58, 0.42])
    
    # Payment Method
    payment_methods = np.random.choice(
        ["UPI", "Credit Card", "Debit Card", "Net Banking", "Cash on Delivery"],
        size=n_samples,
        p=[0.38, 0.25, 0.17, 0.11, 0.09]
    )
    
    # Website Experience Rating (1 to 5)
    website_ratings = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.08, 0.14, 0.32, 0.28, 0.18])
    
    # Return requested
    # Returns correlate slightly with dissatisfaction, apparel, or electronics
    return_requested = []
    for cat in product_categories:
        prob = 0.20 if cat in ["Fashion", "Electronics"] else 0.09
        return_requested.append(np.random.choice(["Yes", "No"], p=[prob, 1 - prob]))
    return_requested = np.array(return_requested)
    
    # Order Frequency
    order_frequencies = np.random.choice(
        ["Rarely", "Monthly", "Bi-weekly", "Weekly"],
        size=n_samples,
        p=[0.28, 0.42, 0.18, 0.12]
    )
    
    # 6. Realistic Satisfaction Score Calculation
    # Formula based on empirical business customer satisfaction drivers:
    # + Positive: high website rating, good service, fast delivery, repeat customer, discount
    # - Negative: slow delivery, returns requested, poor service, low website rating
    latent_score = np.zeros(n_samples)
    
    for i in range(n_samples):
        score = 50.0  # Baseline neutral
        
        # Website Rating impact (+- 20)
        score += (website_ratings[i] - 3) * 7.5
        
        # Service Interaction impact (+- 18)
        score += (service_interactions[i] - 3) * 6.5
        
        # Delivery time penalty: <= 3 days = +10, 4-6 days = +2, 7+ days = heavy penalty
        if delivery_times[i] <= 3:
            score += 10.0
        elif delivery_times[i] <= 5:
            score += 3.0
        elif delivery_times[i] <= 8:
            score -= 8.0
        else:
            score -= 22.0
            
        # Return requested penalty
        if return_requested[i] == "Yes":
            score -= 14.0
            
        # Loyalty effect: repeat buyers are generally more forgiving or satisfied
        if previous_orders[i] >= 8:
            score += 6.0
        elif previous_orders[i] >= 3:
            score += 3.0
        elif previous_orders[i] == 0:
            score -= 2.0
            
        # Discount satisfaction boost
        if discounts_used[i] == "Yes":
            score += 3.5
            
        # Slight category nuances (e.g. COD with delayed delivery causes frustration)
        if payment_methods[i] == "Cash on Delivery" and delivery_times[i] >= 7:
            score -= 5.0
            
        # Natural customer sentiment variance (Gaussian noise)
        noise = np.random.normal(0, 5.0)
        score += noise
        latent_score[i] = score
        
    # Categorize into Low, Medium, High based on thresholds
    # Target distribution approx: ~30% Low, ~35% Medium, ~35% High
    p33 = np.percentile(latent_score, 32)
    p68 = np.percentile(latent_score, 66)
    
    satisfaction_labels = []
    for sc in latent_score:
        if sc <= p33:
            satisfaction_labels.append("Low")
        elif sc <= p68:
            satisfaction_labels.append("Medium")
        else:
            satisfaction_labels.append("High")
            
    # Assemble DataFrame
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "customer_age": ages,
        "gender": genders,
        "product_category": product_categories,
        "purchase_amount": purchase_amounts,
        "delivery_time": delivery_times,
        "service_interaction": service_interactions,
        "previous_orders": previous_orders,
        "discount_used": discounts_used,
        "payment_method": payment_methods,
        "website_rating": website_ratings,
        "return_requested": return_requested,
        "order_frequency": order_frequencies,
        "customer_satisfaction": satisfaction_labels
    })
    
    return df

def main():
    # Set directories relative to repository root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    data_dir = os.path.join(project_root, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    csv_path = os.path.join(data_dir, "customer_satisfaction.csv")
    
    print("=" * 70)
    print("SUBATHRA D - E-Commerce Customer Satisfaction Dataset Generator")
    print("=" * 70)
    print("[*] Generating 1,000 realistic e-commerce customer records...")
    
    df = generate_customer_satisfaction_data(n_samples=1000, random_seed=42)
    df.to_csv(csv_path, index=False)
    
    print(f"[SUCCESS] Dataset successfully saved to: {csv_path}")
    print(f"[INFO] Dataset Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\n[INFO] Target Column Distribution:")
    counts = df["customer_satisfaction"].value_counts()
    for cat, count in counts.items():
        pct = (count / len(df)) * 100
        print(f"    - {cat:7s}: {count:4d} records ({pct:.1f}%)")
        
    print("\n[INFO] First 3 Sample Records:")
    print(df.head(3).to_string(index=False))
    print("=" * 70)

if __name__ == "__main__":
    main()
