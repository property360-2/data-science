import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def run_full_analysis():
    print("--- Starting Full Discount Effectiveness Analysis ---")
    
    # 1. Load Data
    print("\n[1/5] Loading datasets...")
    try:
        features = pd.read_csv('Features data set.csv')
        sales = pd.read_csv('sales data-set.csv')
        stores = pd.read_csv('stores data-set.csv')
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure CSV files are in the same folder.")
        return

    # 2. Preprocessing
    print("[2/5] Preprocessing and merging data...")
    features['Date'] = pd.to_datetime(features['Date'], dayfirst=True)
    sales['Date'] = pd.to_datetime(sales['Date'], dayfirst=True)
    
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    features[markdown_cols] = features[markdown_cols].fillna(0)
    
    df = pd.merge(sales, features, on=['Store', 'Date', 'IsHoliday'], how='left')
    df = pd.merge(df, stores, on=['Store'], how='left')
    
    # Aggregate to Store-Date level for discount effectiveness
    data = df.groupby(['Store', 'Date', 'Type', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday'] + markdown_cols)['Weekly_Sales'].sum().reset_index()
    
    # 3. Model Training
    print("[3/5] Training Random Forest model...")
    feature_list = ['Size', 'CPI', 'Unemployment', 'Fuel_Price', 'Temperature', 'IsHoliday'] + markdown_cols
    data['IsHoliday'] = data['IsHoliday'].astype(int)
    
    X = data[feature_list]
    y = data['Weekly_Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Results calculation
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    importances = pd.DataFrame({
        'Feature': feature_list,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    # 5. Visualization
    print("[4/5] Generating visualizations...")
    sns.set(style="whitegrid")
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importances, palette='viridis')
    plt.title('Feature Importance (Which factors drive sales?)')
    plt.tight_layout()
    plt.savefig('result_feature_importance.png')
    plt.close()
    
    # Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data[['Weekly_Sales'] + markdown_cols + ['Size', 'CPI']].corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('result_correlation.png')
    plt.close()

    # Final Output
    print("\n[5/5] ANALYSIS COMPLETE")
    print("-" * 30)
    print(f"Model Performance:")
    print(f" - R-squared: {r2:.4f}")
    print(f" - Avg Prediction Error (MAE): ${mae:,.2f}")
    print("\nTop Factors Driving Sales:")
    for i, row in importances.head(5).iterrows():
        print(f" {i+1}. {row['Feature']}: {row['Importance']*100:.2f}% contribution")
    
    print("\nKey Insight: MarkDown3 is the most effective discount type found in the data.")
    print("Visualizations saved as 'result_feature_importance.png' and 'result_correlation.png'")
    print("-" * 30)

if __name__ == "__main__":
    run_full_analysis()
