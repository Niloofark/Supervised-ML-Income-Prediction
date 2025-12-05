## **Predicting Income levels Using Supervised Machine Learning Algorithms: A Study on the United States Census 1994 Dataset**

This project predicts individual income levels using socio-economic features from the U.S. Census 1994 dataset, classifying whether a person earns ≤ $50K or > $50K per year. The workflow includes preprocessing, EDA, training eight ML models, and comparing their performance.

The results show strong patterns in income distribution bias and gender inequality. XGBoost delivers the best performance, followed closely by Random Forest and Gradient Boosting. Models like Decision Tree, Neural Network, and SVM perform less effectively, partly due to class imbalance.

All high- and low-performing models, along with detailed evaluation metrics, are documented in separate Jupyter notebooks in the Niloofar_FinalProject repository.
**HighPerformance_Models.ipynb**
1. Logistic Regression
2. Random Forest
3. Gradient Boosting
4. K-Nearest Neighbor (K-NN)
5. XGBoost

**LowPerformance_Models.ipynb**
6. Decision Tree
7. Neural Network (Deep Learning)
8. Support Vector Machine (SVM)

## Installation Instructions

```bash
# Clone the repository
git clone https://github.com/Niloofark/Niloofar_FinalProject.git

# Navigate to the project
cd Niloofar_FinalProject

# Install required libraries
pip install numpy pandas scikit-learn matplotlib seaborn xgboost tensorflow

# Run the project
jupyter notebook HighPerformance_Models.ipynb
```

**Data**
Source: The dataset is sourced from the UC Irvine Machine Learning Repository and Kaggle, containing 48,842 instances of data with 15 features related to demographic and socio-economic information.
Features: Includes age, work class, education, marital status, occupation, relationship, race, gender, capital gain, capital loss, hours per week, and native country, among others.
Preprocessing: Handled missing values by replacing them with 'Unknown', removed duplicates, and standardized continuous features. Capital gain and loss were dropped due to skewness and low correlation with income.

**Evaluation**
Metrics: Used a combination of quantitative metrics (accuracy, precision, recall, F1-score, ROC-AUC, cross-validation score, and confusion matrix) and qualitative metrics (model interpretability, robustness, feature importance, error analysis).
Results: XGBoost showed the highest performance across most metrics, followed by Logistic Regression and Gradient Boosting.

**Results**
Summary: XGBoost had the best overall performance, balancing accuracy and training time. Logistic Regression was the most interpretable, while Gradient Boosting also performed well.
Insights: The study revealed biases in income distribution, particularly gender inequality, and identified significant socio-economic factors affecting income levels.


Contributions are welcome. Please submit pull requests or issues with clear descriptions. Ensure your contributions align with the project's goals and standards.
Please contact me: karimi.ni@northeastern.edu
