# Discount Effectiveness Analysis (4-2 Alvior)

## 1. Project Title & Objective
**Objective:** Determine which discounts drive sales.
The project aims to analyze the impact of various discount strategies (MarkDown1 through MarkDown5) on weekly sales performance within a retail context, while accounting for store characteristics and environmental factors.

---

## 2. Dataset
- **Source:** Kaggle "Retail Data Analytics"
- **Size:** Multiple CSV files (Sales: ~13MB, Features: ~600KB, Stores: ~1KB)
- **Features:**
    - **Numerical:** `Weekly_Sales`, `Temperature`, `Fuel_Price`, `CPI`, `Unemployment`, `Size`, `MarkDown1-5`.
    - **Categorical:** `Store`, `Type`, `IsHoliday`.
    - **Time-based:** `Date`.

---

## 3. Data Preprocessing
The preprocessing was handled in [preprocess.py](./preprocess.py) and consolidated in [full_discount_analysis.py](./full_discount_analysis.py):
1. **Handling Missing Data:** MarkDown columns (1-5) had significant null values. These were filled with `0` under the assumption that a null value indicates no discount was applied.
2. **Data Integration:** Merged three datasets (`sales`, `features`, and `stores`) using `Store` and `Date` as primary keys.
3. **Date Conversion:** Converted text-based dates into standard Python `datetime` objects for time-series alignment.
4. **Aggregation:** Data was aggregated to the Store-Date level to align markdown influences with total weekly store performance.

---

## 4. Data Mining Techniques / Modeling
- **Technique:** Regression / Importance Analysis
- **Algorithm:** **Random Forest Regressor** (scikit-learn)
- **Rationale:** Random Forest was chosen for its ability to handle non-linear relationships and its robust "Feature Importance" output, which directly answers which factors "drive" sales.
- **Tools:** Use `pandas` for data manipulation, `scikit-learn` for modeling, and `seaborn` for visualization.

---

## 5. SWOT Analysis

| **Strengths** | **Weaknesses** |
| :--- | :--- |
| - High Model Performance (R² = 0.94).<br>- Clear hierarchical identification of drivers. | - Heavily missing MarkDown data required zero-filling.<br>- Store size dominates other features. |
| **Opportunities** | **Threats** |
| - Prioritize **MarkDown3** strategies.<br>- Optimize inventory based on CPI/Unemployment trends. | - Economic volatility (CPI) remains a major external risk.<br>- Customer behavior shifts over time. |

---

## 6. Visual Presentation
The analysis generated two key charts using `matplotlib` and `seaborn`:

### **Feature Importance**
Created using `sns.barplot` on the `model.feature_importances_` array. It shows how much each variable contributed to the prediction.
![Feature Importance Chart](./result_feature_importance.png)

### **Correlation Heatmap**
Created using `df.corr()` and `sns.heatmap` to show direct linear relationships between sales and discounts.
![Correlation Heatmap](./result_correlation.png)

---

## 7. Results & Interpretation
### **Patterns & Predictions**
- **The dominator**: **Store Size** is the #1 predictor (80.08% importance). Simply put, larger stores sell more.
- **The most effective discount**: **MarkDown3** (2.31%) outperformed all other discounts. 
- **Macro-Environmental Impact**: CPI and Unemployment together influence sales by over **13%**, showing that the economy is a bigger driver than individual discounts.

### **Metrics**
- **R-squared (R²):** **0.9406** (Great fit)
- **Mean Absolute Error (MAE):** **$76,507.56**

---

## 8. Recommendations / Actionable Insights
- **Focus Marketing**: Redirect promotional budgets toward the activities categorized under "MarkDown3".
- **Store-Specific Strategy**: Since Size is so influential, larger stores should be the primary targets for high-volume markdown events.
- **Economic Agility**: Monitor CPI and Unemployment as primary "threat" indicators for future sales forecasts.

---

## 9. Optional Extensions
- **Predictive Modeling**: Future work can involve Time-Series forecasting (ARIMA/Prophet) to predict sales 12 weeks in advance based on planned discounts.
- **Anomaly Detection**: Investigating stores that underperform despite high MarkDown3 spending.

---
### **How to Run the full analysis:**
Run the consolidated script [full_discount_analysis.py](./full_discount_analysis.py) to regenerate these results and charts automatically.
