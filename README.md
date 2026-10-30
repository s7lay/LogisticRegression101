# Bank Customer Subscription Prediction

This project builds a predictive model to forecast whether a bank customer will subscribe to a campaign or product (`subscribed`), using machine learning techniques in Python.

## Dataset
- **File:** `6-bank_customers.csv`[cite: 1]
- **Features:** Includes demographic information, financial balance, job satisfaction, call durations, campaign response scores, and loan statuses[cite: 1].
- **Target Variable:** `subscribed` (Binary classification: 0 or 1)[cite: 1].

## Workflow & Methodology
1. **Data Preprocessing:** Loaded the dataset using `pandas`, inspected data types/statistics, and separated features (`X`) from the target variable (`y`)[cite: 1].
2. **Train-Test Split:** Divided the dataset into training and testing sets (70% train, 30% test) using `scikit-learn`[cite: 1].
3. **Baseline Model:** Trained an initial **Logistic Regression** classifier and evaluated it using accuracy score, classification report, and confusion matrix[cite: 1].
4. **Hyperparameter Tuning:** 
   - Performed **GridSearchCV** with `StratifiedKFold` cross-validation to find the optimal hyperparameters (`C`, `penalty`, `solver`)[cite: 1].
   - Experimented with **RandomizedSearchCV** for efficient hyperparameter optimization[cite: 1].

## Technologies & Libraries Used
- **Python**[cite: 1]
- **Pandas & NumPy** (Data manipulation and analysis)[cite: 1]
- **Matplotlib & Seaborn** (Data visualization)[cite: 1]
- **Scikit-Learn** (Machine learning models, metrics, and hyperparameter tuning)[cite: 1]

## Results
- The optimized Logistic Regression model achieved an accuracy of approximately **92%** on the test dataset[cite: 1].
