# 🩺 Diabetes Risk Analysis using EDA

This project focuses on **exploratory data analysis (EDA)** of the **PIMA Indians Diabetes Dataset** to identify key health indicators and their correlation with the presence of diabetes.

---

## 📂 Dataset Info

- 📁 Source: [Kaggle - PIMA Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- 🔢 Rows: 768 | 📊 Columns: 9
- 🎯 Target Column: `Outcome` (1 = Diabetes, 0 = No Diabetes)

---

## 🎯 Objective

- Understand the relationship between:
  - Glucose level and age
  - BMI patterns and diabetes
  - Other health indicators like insulin, blood pressure, etc.
- Perform correlation analysis using heatmaps
- Identify high-risk groups visually using data plots

---

## 🛠️ Tools & Libraries

- `pandas`
- `matplotlib`
- `seaborn`
- `numpy`
- `jupyter notebook` / `VS Code`

---

## 📊 Key Visualizations

- ✅ Glucose vs Age scatterplot (colored by diabetes status)
- ✅ BMI distribution histogram
- ✅ Correlation heatmap
- ✅ Outcome (0/1) count plot

---

## 🔍 Insights

- Higher glucose levels are strongly associated with diabetes
- Most diabetic individuals fall within a specific BMI and age range
- Features like insulin and skin thickness show weak correlation individually

---

## 🚀 How to Run

```bash
pip install pandas matplotlib seaborn jupyter
