# Retail Data Analytics: Discount Effectiveness Analysis

## Overview
This project analyzes the "Retail Data Analytics" dataset to determine which promotional markdowns (MarkDown 1-5) are most effective in driving weekly sales. 

## Project Objective
- Identify the primary drivers of retail sales.
- Quantify the impact of promotional discounts on revenue.
- Provide actionable business recommendations based on statistical importance.

## Key Findings
- **Model Accuracy**: R-squared of **0.9406**.
- **Top Driver**: **Store Size** (80.08% contribution).
- **Most Effective Discount**: **MarkDown3** (2.31% contribution).

## File Structure
- `full_discount_analysis.py`: Unified script that handles loading, preprocessing, and analysis.
- `Discount_Analysis_Report.md`: A comprehensive documentation of the project objective, methodology, SWOT analysis, and results.
- `analysis_results.txt`: Raw statistical output from the model.
- `preprocess.py`: Original script for data merging and cleaning.
- `analysis.py`: Original script for modeling and plot generation.
- `result_feature_importance.png`: Visualization of the top factors affecting sales.
- `result_correlation.png`: Heatmap showing relationships between variables.

## How to Run
1. Ensure you have the datasets in the root folder:
   - `sales data-set.csv`
   - `Features data set.csv`
   - `stores data-set.csv`
2. Run the unified analysis script:
   ```bash
   python full_discount_analysis.py
   ```
3. Check the terminal for summary results and the root folder for generated PNG charts.

---
*Created by: 4-2 Alvior*
