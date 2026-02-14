import pandas as pd
import numpy as np

def preprocess_data():
    # Load datasets
    features = pd.read_csv('Features data set.csv')
    sales = pd.read_csv('sales data-set.csv')
    stores = pd.read_csv('stores data-set.csv')

    # Convert Date to datetime for features and sales
    features['Date'] = pd.to_datetime(features['Date'], dayfirst=True)
    sales['Date'] = pd.to_datetime(sales['Date'], dayfirst=True)

    # Fill NaN MarkDowns with 0
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    features[markdown_cols] = features[markdown_cols].fillna(0)

    # Merge sales and features on Store, Date, and IsHoliday
    df = pd.merge(sales, features, on=['Store', 'Date', 'IsHoliday'], how='left')

    # Merge with stores data
    df = pd.merge(df, stores, on=['Store'], how='left')

    # Aggregating sales by Store and Date (since MarkDowns are store-level)
    # This avoids over-representing discounts if we analyze at the Dept level
    # however, for a comprehensive analysis, we might keep Dept or aggregate.
    # The objective is "Determine which discounts drive sales"
    
    # Store-level analysis might be better for MarkDown effectiveness
    store_date_sales = df.groupby(['Store', 'Date', 'Type', 'Size', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'IsHoliday'] + markdown_cols)['Weekly_Sales'].sum().reset_index()

    # Save processed data
    store_date_sales.to_csv('processed_retail_data.csv', index=False)
    print("Preprocessing complete. Saved to processed_retail_data.csv")
    print(f"Total records: {len(store_date_sales)}")

if __name__ == "__main__":
    preprocess_data()
