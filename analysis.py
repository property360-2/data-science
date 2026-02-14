import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

def perform_analysis():
    # Load processed data
    df = pd.read_csv('processed_retail_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 1. Correlation Analysis
    cols_to_corr = ['Weekly_Sales', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size']
    corr_matrix = df[cols_to_corr].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap: Sales and Discounts')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.close()
    
    # 2. Time Series Sales and Markdowns
    df_sorted = df.sort_values('Date')
    df_weekly = df_sorted.groupby('Date')[['Weekly_Sales', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].mean().reset_index()
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_weekly['Date'], df_weekly['Weekly_Sales'], label='Avg Weekly Sales', color='blue', linewidth=2)
    plt.title('Average Weekly Sales Over Time')
    plt.ylabel('Sales')
    plt.legend()
    plt.savefig('sales_trends.png')
    plt.close()

    # 3. Random Forest for Feature Importance
    # We want to see how much each markdown contributes to predicting sales
    features = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Size', 'IsHoliday']
    # Encode IsHoliday
    df['IsHoliday'] = df['IsHoliday'].astype(int)
    
    X = df[features]
    y = df['Weekly_Sales']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance for Weekly Sales')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    # 4. Model Evaluation
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Write summary to a file
    with open('analysis_results.txt', 'w') as f:
        f.write("--- Analysis Results ---\n")
        f.write(f"Model R-squared: {r2:.4f}\n")
        f.write(f"Model MAE: {mae:.2f}\n\n")
        f.write("Correlations with Weekly_Sales:\n")
        f.write(corr_matrix['Weekly_Sales'].sort_values(ascending=False).to_string())
        f.write("\n\nFeature Importance:\n")
        f.write(feature_importance_df.to_string(index=False))
        
    print("Analysis complete. Results and plots saved.")

if __name__ == "__main__":
    perform_analysis()
