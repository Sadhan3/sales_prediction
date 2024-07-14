# sales_prediction


This Python script performs a machine learning task to predict sales categories (low, medium, high) based on advertising expenditures (TV, Radio, Newspaper) using a RandomForestClassifier. Here's a step-by-step breakdown:

1. **Data Loading and Exploration:**
   - The script begins by loading a dataset (`Advertising.csv`) from a specified URL into a Pandas DataFrame (`data`).
   - It checks for any missing values in the dataset using `data.isnull().sum()`.

2. **Data Preprocessing:**
   - It bins the 'Sales' column into categorical labels:
     - 'Low' for sales values between 0 and 10.
     - 'Medium' for sales values between 10 and 20.
     - 'High' for sales values above 20.
   - These categories are encoded into integers (`0` for 'Low', `1` for 'Medium', `2` for 'High') using `pd.cut()` and `astype(int)`.

3. **Data Splitting:**
   - The dataset is split into training (`X_train`, `y_train`) and testing (`X_test`, `y_test`) sets using `train_test_split()` from `sklearn.model_selection`.
   - The training set comprises 80% of the data, and the testing set comprises 20%.

4. **Model Training:**
   - A RandomForestClassifier model is initialized with `random_state=42` for reproducibility.
   - The model is trained on the training data (`X_train`, `y_train`) using `model.fit()`.

5. **Model Evaluation:**
   - The trained model predicts sales categories (`y_pred`) for the test set (`X_test`).
   - Evaluation metrics are computed:
     - Confusion Matrix (`cm`) using `confusion_matrix()`.
     - Classification Report (`cr`) using `classification_report()`, providing metrics like precision, recall, and F1-score.
     - Precision, Recall, and F1 Score are calculated using `precision_score()`, `recall_score()`, and `f1_score()` respectively.

6. **Visualization:**
   - The confusion matrix (`cm`) is visualized as a heatmap using `matplotlib` and `seaborn`.
   - The heatmap helps visualize the model's performance in predicting actual versus predicted sales categories.

7. **Making Predictions:**
   - Predictions are made for new advertising expenditure data (`new_data`) using the trained model.
   - The predicted sales categories for `new_data` are printed to the console.

### Goal of the Code

The goal of this code is to utilize machine learning techniques to:
- Predict sales categories (low, medium, high) based on advertising expenditures (TV, Radio, Newspaper).
- Evaluate the performance of the RandomForestClassifier model using a confusion matrix and classification report.
- Provide predictions for new data points based on the trained model.

This script demonstrates a complete pipeline from data loading and preprocessing to model training, evaluation, and prediction, focusing on solving a classification problem in the context of advertising and sales analysis.
