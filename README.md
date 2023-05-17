# ⚙️🏭Failure Machine Prediction
A classification model that predicts a machine failure types. 

Built in:
- Model: Support Vector Machine (SVM).
- Feature outsampling: Synthetic Minority Oversampling Technique (SMOTE).
- Cross validation: Stratified KFolds, GridSearchCV.
- Model evaluation: Precision/Recall, F1 Score, Confusion Matrix, Accuracy.

These are possible machine failures:

- No Failure
- Heat Dissipation Failure
- Power Failure
- Overstain Failure
- Tool Wear Failure
- Random Failures

<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/training%20confusion%20matrix.png?raw=true" alt="alt text" width="350" height="270"> <img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/test%20confusion%20matrix.png?raw=true" alt="alt text" width="350" height="270"> 

Model features:
- Type: L, M, H
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

## 📑Data Acquisition
Dataset can be obtained from Kaggle: https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

## 🔍Exploratory Data Analysis (EDA)
This repository showcases the exploratory data analysis (EDA). The combination of these techniques enables to gain a comprehensive understanding of the dataset and provided valuable insights for further analysis and modeling such as: 

- Identify missing or incomplete data
- Understanding the level of uniqueness and variability in the dataset
- Observing the relationships and potential correlations between different features, uncover patterns, identify outliers, and assess the overall shape of the data.

Observation: 
- Column ['UDI', 'Product ID', 'Target'] considered not needed.
- Features ['Type','Failure Type'] needed to be encoded.
- The target label ['Failure Type'] is imbalanced with 96% is No Failure.

<p align="left">
  <img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/eda%20-%20histogram.png?raw=true" width="500" height="300" alt="Image 1">
  <img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/eda%20-%20pairplot.png?raw=true" width="500" height="500" alt="Image 2">
</p>

## 🧰Feature Engineering
### 🧩Feature Selection
Found during EDA that column ['UDI', 'Product ID', 'Target'] considered not needed in features. The column are dropped from the dataset.

<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/feature%20selection%20-%20drop%20columns.png?raw=true" width="700" height="250" alt="alt text">

### 🔢Feature and Target Encoding
Feature ['Type','Failure Type'] are in a string type. Feature encoding can be performed using ``import category_encoders as ce`` library to proceed the modeling process. ``ce.OrdinalEncoder(cols=['column'])``, ``fit_transform(df)``. Below is the feature encoding result.

<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/feature%20encoding%20-%20machine%20type,%20failure%20type.png?raw=true" width="700" height="250" alt="alt text">

### 🧪Feature Oversampling
The target label found inbalanced in the dataset, where 96% target label are **No Failure**. Unbalanced dataset may result to an overfitting model towards a specific label that does not generalized well towards other labels. Synthetic Minority Oversampling Technique (SMOTE) will be proceed next to oversampling the features, so the target label is balanced. 

SMOTE can be intialized by importing ``from imblearn.combine import SMOTETomek`` and resample data using ``.fit_resample(df.values,y)``. Below is the result of oversampling data vs original data.

<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/feature%20oversampling%20-%20%20target%20value%20count.png?raw=true" width="700" height="200" alt="alt text">
<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/feature%20oversampling%20-%20smote%20shape.png?raw=true" width="700" height="200" alt="alt text">

### 🎛️Feature Scaling
Since the features from the dataset have a different range, feature scaling will be perform to improve model convergence, model enhancing, standardize interpretation using ``from sklearn.preprocessing import StandardScaler``,``.fit_transform(feature)``. Below is the feature scaling result.

<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/feature%20scaling%20.png?raw=true" width="700" height="200" alt="alt text">

### 💠Feature Split
The datasets will be splitted into partitions for further model evaluation, fine-tuning, preventing overfitting model and well assessed generalization model. The data will be splitted into:

- Training set (70%)
- Test set (30%)

The feature split can be performed using ``from sklearn.model_selection import train_test_split``, ``train_test_split(feature, target, test_size=0.3)``

## 🏂Cross Validation
Cross validation are performed to assess the performance and generalization ability of a model, which performing model training and evaluation iteratively, and then aggregating the results.

The cross validation technique that will be used is Stratified KFolds with GridSearchCV, to find the best hyperparameters for the classification task. The dataset will be divided into k folds ensuring that the class distribution is preserved in each fold.

Then the performance of the model is evaluated on the validation set (1 fold) using a chosen evaluation metric (e.g., accuracy, F1 score). The average performance across all k folds is calculated. The combination of hyperparameters that yields the best average performance across the k folds is selected as the optimal set of hyperparameters.

K will be set = 5. ``skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)``

Here are the model and variations of hyperparameters for cross validation:<br>
``model = svm.SVC(gamma='auto')``<br>
``params = {'C': [1, 10, 15, 30],'kernel': ['rbf', 'poly']}``

Cross validaition with StratifiedKFold and GridSearchCV initialize and training:<br>
``clf = GridSearchCV(model, params, cv=skf, return_train_score=True)``<br>
``clf.fit(xtrain, ytrain)``

Getting the best parameters and model after cross validation:<br>
``clf.best_params_``<br>
``clf.best_estimator_``

Result:
The best parameters are `{'C': 30, 'kernel': 'rbf'}`

## 🎿Model Evaluation
After getting the best model, the model will be evaluated back for every data splits including training, test and all dataset. During model evaluation, the best model prediction, classification report, and confusion matrix will be generated for further observation. 

Below is the model evaluation result:
- Training set:

```
              precision    recall  f1-score   support
              
           1       0.97      0.85      0.91      6758
           2       1.00      1.00      1.00      6694
           3       0.96      0.99      0.97      6724
           4       0.99      0.99      0.99      6871
           5       0.91      0.97      0.94      6788
           6       0.97      1.00      0.98      6688

    accuracy                           0.97     40523
   macro avg       0.97      0.97      0.97     40523
weighted avg       0.97      0.97      0.97     40523
```
<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/training%20confusion%20matrix.png?raw=true" width="300" height="300" alt="alt text">

- Testing set: 

```
              precision    recall  f1-score   support

           1       0.96      0.84      0.90      2883
           2       1.00      1.00      1.00      2951
           3       0.96      0.99      0.97      2924
           4       0.99      0.99      0.99      2781
           5       0.91      0.96      0.93      2864
           6       0.97      1.00      0.98      2964

    accuracy                           0.96     17367
   macro avg       0.96      0.96      0.96     17367
weighted avg       0.96      0.96      0.96     17367
```
<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/test%20confusion%20matrix.png?raw=true" width="300" height="300" alt="alt text">






