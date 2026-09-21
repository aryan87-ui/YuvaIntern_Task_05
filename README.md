# YuvaIntern_Task_05
# YuvaIntern Task 05 – Machine Learning Algorithms and Model Evaluation

## 📌 About the Project

This project is part of the **Yuva Intern Data Analytics Internship – Week 5** task.

The objective of this task is to transition from descriptive and statistical analysis to **predictive analytics using Machine Learning**. The project uses the **Sample - Superstore** dataset to build regression models for predicting sales and to evaluate their performance using appropriate machine learning metrics.

The complete workflow includes data preprocessing, feature selection, model training, cross-validation, hyperparameter tuning, model evaluation, and visualization.

---

## 🎯 Objectives

The main objectives of this task are:

* Implement Machine Learning algorithms using Python.
* Formulate a suitable regression problem.
* Preprocess numerical and categorical features.
* Train and compare multiple regression models.
* Perform cross-validation.
* Tune model hyperparameters using GridSearchCV.
* Evaluate model performance using MAE, MSE, RMSE, and R².
* Generate visualizations to understand model performance.
* Document the complete Machine Learning workflow and findings.

---

## 📊 Dataset

The project uses the **Sample - Superstore.csv** dataset.

### Dataset Information

* **Total Rows:** 9,994
* **Total Columns:** 21
* **Target Variable:** `Sales`
* **Problem Type:** Regression

### Important Columns

* Order Date
* Ship Mode
* Segment
* Region
* Category
* Sub-Category
* Quantity
* Discount
* Sales
* Profit

The dataset contains information about orders, customers, products, regions, sales, quantities, discounts, and profits.

---

## 🧩 Problem Statement

The objective is to predict the **Sales** value of a transaction using selected categorical, numerical, and date-based features.

The selected features are:

```text
Ship Mode
Segment
Region
Category
Sub-Category
Quantity
Discount
Order_Year
Order_Month
```

The target variable is:

```text
Sales
```

---

## 🔄 Machine Learning Workflow

The following workflow was implemented:

```text
Dataset Loading
      ↓
Data Inspection
      ↓
Missing Value Check
      ↓
Date Feature Engineering
      ↓
Feature Selection
      ↓
Train-Test Split
      ↓
Data Preprocessing
      ↓
Linear Regression
      ↓
Random Forest Regression
      ↓
5-Fold Cross-Validation
      ↓
Hyperparameter Tuning
      ↓
GridSearchCV
      ↓
Tuned Random Forest
      ↓
Model Evaluation
      ↓
Visualization
      ↓
Insights and Conclusion
```

---

## 🧹 Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Checked the dataset dimensions and column names.
3. Checked for missing values.
4. Converted `Order Date` into datetime format.
5. Extracted:

   * `Order_Year`
   * `Order_Month`
6. Selected relevant features.
7. Separated features and target variable.
8. Split the data into training and testing sets using an 80:20 ratio.
9. Applied median imputation to numerical features.
10. Applied most-frequent imputation to categorical features.
11. Applied One-Hot Encoding to categorical variables.

No missing values were found in the original dataset.

---

## 🤖 Machine Learning Algorithms

### 1. Linear Regression

Linear Regression was implemented as the baseline model.

It was selected because it is:

* Simple and interpretable.
* Computationally efficient.
* Suitable as a baseline for regression problems.
* Useful for comparing the performance of more complex models.

### 2. Random Forest Regressor

Random Forest Regression was implemented to capture potentially nonlinear relationships in the dataset.

It was selected because:

* It can model nonlinear relationships.
* It can capture interactions between features.
* It is an ensemble-based approach.
* It generally provides more flexibility than a simple linear model.

---

## 🔁 Train-Test Split

The dataset was divided into:

| Dataset      | Records |
| ------------ | ------: |
| Training Set |   7,995 |
| Testing Set  |   1,999 |

The split ratio was:

```text
80% Training
20% Testing
```

A `random_state` of `42` was used to make the experiment reproducible.

---

## 🔬 Cross-Validation

5-Fold K-Fold Cross-Validation was performed on the training data.

### Cross-Validation R² Scores

```text
0.0807
0.2784
0.0232
-0.1833
0.1368
```

### Mean Cross-Validation R²

```text
0.0671
```

The variation between folds indicates that model performance differs across validation subsets. This observation is considered when interpreting the model's generalization capability.

---

## ⚙️ Hyperparameter Tuning

`GridSearchCV` was used to optimize the Random Forest model.

The parameter search tested:

```text
n_estimators
max_depth
min_samples_split
min_samples_leaf
```

A total of:

```text
24 parameter combinations
×
3-fold cross-validation
=
72 fits
```

were evaluated.

### Best Parameters

```text
n_estimators = 200
max_depth = 10
min_samples_split = 5
min_samples_leaf = 2
```

### Best Cross-Validation RMSE

```text
527.4981
```

---

## 📈 Model Evaluation

The models were evaluated using:

### MAE – Mean Absolute Error

Measures the average absolute difference between actual and predicted values.

### MSE – Mean Squared Error

Squares the prediction errors, giving larger errors greater influence.

### RMSE – Root Mean Squared Error

Measures prediction error in the same unit as the target variable.

### R² – R-Squared

Measures the proportion of variation in the target explained by the model on the evaluated data.

---

## 📊 Model Performance

The final test-set results were:

| Model                   |        MAE |       RMSE |         R² |
| ----------------------- | ---------: | ---------: | ---------: |
| Linear Regression       |     226.36 |     693.34 |     0.1862 |
| Random Forest           |     215.65 |     684.43 |     0.2070 |
| **Tuned Random Forest** | **196.74** | **667.62** | **0.2454** |

The tuned Random Forest produced the lowest MAE and RMSE and the highest R² among the three evaluated models on the held-out test set.

---

## 📉 Visualizations

The project generates the following visualizations:

### Actual vs Predicted

Shows the relationship between actual Sales values and the predictions generated by the tuned model.

### Residual Plot

Shows prediction residuals against predicted Sales values and helps identify systematic prediction patterns.

### Model Comparison

Compares the performance of the evaluated models using RMSE.

---

## 📁 Project Structure

```text
YuvaIntern_Task_05/
│
├── dataset/
│   └── Sample - Superstore.csv
│
├── src/
│   └── machine_learning.py
│
├── output/
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   ├── model_comparison.png
│   ├── model_comparison.csv
│   ├── predictions.csv
│   └── tuning_results.csv
│
├── report/
│   └── Week_5_Machine_Learning_Report.docx
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* GridSearchCV
* K-Fold Cross-Validation

---

## ▶️ How to Run the Project

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

### 2. Navigate to the project directory

```bash
cd YuvaIntern_Task_05
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

#### Windows PowerShell

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell execution policy prevents activation, the Python executable inside `.venv` can also be used directly.

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Machine Learning script

```bash
python src/machine_learning.py
```

The generated model evaluation files and visualizations will be saved in the `output` folder.

---

## 📌 Key Findings

* The dataset contained **9,994 records and 21 columns**.
* No missing values were found in the dataset.
* Random Forest performed better than Linear Regression on the held-out test set.
* Hyperparameter tuning improved the Random Forest test-set results.
* The tuned Random Forest achieved:

  * **MAE:** 196.74
  * **RMSE:** 667.62
  * **R²:** 0.2454
* Cross-validation showed noticeable variation between folds, indicating that model performance can depend on the validation subset.
* The selected features provide useful predictive information, but they do not explain all variation in Sales.

---

## ⚠️ Limitations

* The current feature set is relatively limited.
* Product-level and customer-level information could be explored more deeply.
* Cross-validation results varied considerably across folds.
* Final test metrics are based on one held-out train-test split.
* The current model is an internship-level predictive analysis rather than a production forecasting system.
* The task predicts transaction-level Sales rather than future aggregated sales.

---

## 🚀 Future Improvements

Possible improvements include:

* Add additional date-based features.
* Explore product and customer-level features.
* Investigate interaction features.
* Test additional regression algorithms.
* Perform more robust cross-validation.
* Explore time-aware validation for future sales forecasting.
* Analyze feature importance and permutation importance.
* Investigate transformations for highly skewed Sales values.
* Consider an aggregated/time-series forecasting formulation for future-period sales.

---

## 📄 Documentation

A detailed report is included in the `report` folder.

The report covers:

* Machine Learning methodology
* Data preprocessing
* Algorithm selection
* Model training
* Cross-validation
* Hyperparameter tuning
* Evaluation metrics
* Model comparison
* Visualizations
* Results interpretation
* Limitations
* Future improvements
* Conclusion

---

## ✅ Conclusion

This Week 5 project demonstrates an end-to-end Machine Learning workflow for predictive Sales analysis using Python and Scikit-learn.

The project successfully covers data preprocessing, regression model development, cross-validation, hyperparameter tuning, model evaluation, visualization, and documentation.

The tuned Random Forest provided the strongest held-out test-set metrics among the evaluated models, while the cross-validation results highlighted the importance of further feature engineering and validation in future iterations.

---

## 👨‍💻 Author

**Aryan Verma**

**Yuva Intern – Data Analytics Internship**

---

## 📌 Internship Task

**Week 5: Implementing Machine Learning Algorithms and Model Evaluation**

This project was completed as part of the Yuva Intern Data Analytics Internship.

