# ⚙️🧬Equipment Failure Prediction
A classification model built on Support Vector Machine (SVM), Synthetic Minority Oversampling Technique (SMOTE), Stratified KFolds, GridSearchCV that predicts a machine failures. These are possible machine failures:

- No Failure
- Heat Dissipation Failure
- Power Failure
- Overstain Failure
- Tool Wear Failure
- Random Failures 

Model features: Type (L, M, H), Air Temperature [K], Process Temperature [K], Rotational Speed [rpm], Torque [Nm], Tool Wear [min]

Below is the final results of the model:

<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/training%20confusion%20matrix.png?raw=true" alt="alt text" width="350" height="270"> <img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/test%20confusion%20matrix.png?raw=true" alt="alt text" width="350" height="270"> 

## 📑Data Acquisition
Dataset can be obtained from Kaggle: https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification

## 🔍Exploratory Data Analysis (EDA)
This repository showcases the exploratory data analysis (EDA). The combination of these techniques enables to gain a comprehensive understanding of the dataset and provided valuable insights for further analysis and modeling such as: 

- Identify missing or incomplete data
- Understanding the level of uniqueness and variability in the dataset
- Observing the relationships and potential correlations between different features, uncover patterns, identify outliers, and assess the overall shape of the data.

Observation: 
- Column `df['UDI', 'Product ID', 'Target']` considered not needed.
- Features `df['Type','Failure Type']` needed to be encoded.
- The target label `df['Failure Type']` is imbalanced with **96% is No Failure**.

<p align="left">
  <img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/eda%20-%20histogram.png?raw=true" width="500" height="300" alt="Image 1">
  <img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/eda%20-%20pairplot.png?raw=true" width="500" height="500" alt="Image 2">
</p>

## 🧪Feature Engineering
### 🧩Feature Selection
Found during EDA that column ['UDI', 'Product ID', 'Target'] considered not needed in features. The column are dropped from the dataset.
`df_drop = df.drop(['UDI', 'Product ID', 'Target'], axis=1)`
`df_drop.head()`

```	
	Type	Air temperature [K]	Process temperature [K]	Rotational speed [rpm]	Torque [Nm]	Tool wear [min]	Failure Type
0	  M	          298.1	              308.6	                  1551	              42.8	          0	        No Failure
1	  L	          298.2	              308.7	                  1408	              46.3	          3	        No Failure
2	  L	          298.1	              308.5	                  1498	              49.4	          5	        No Failure
3	  L	          298.2	              308.6	                  1433	              39.5	          7	        No Failure
4	  L	          298.2	              308.7	                  1408	              40.0	          9	        No Failure
```

### 🔢Feature and Target Encoding
Feature ['Type','Failure Type'] are in a string type. Feature encoding can be performed using ``import category_encoders as ce`` library to proceed the modeling process. ``ce.OrdinalEncoder(cols=['column'])``, ``fit_transform(df)``. Below is the feature encoding result.

`
import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=['Type','Failure Type'])
df_encode = encoder.fit_transform(df_drop)
df_encode.head(5)
`
```
      Type	Air temperature [K]	Process temperature [K]	Rotational speed [rpm]	Torque [Nm]	Tool wear [min]	Failure Type
0	  1	        298.1	                308.6	                1551	                42.8	          0	          1
1	  2	        298.2	                308.7	                1408	                46.3	          3	          1
2	  2	        298.1	                308.5	                1498	                49.4	          5	          1
3	  2	        298.2	                308.6	                1433	                39.5	          7	          1
4	  2	        298.2	                308.7	                1408	                40.0	          9	          1
```
### 🏔️Feature Oversampling
The target label found inbalanced in the dataset, where 96% target label are **No Failure**. Unbalanced dataset may result to an overfitting model towards a specific label that does not generalized well towards other labels. 

Synthetic Minority Oversampling Technique (SMOTE) will be proceed next to oversampling the features, so the target label is balanced. 

SMOTE can be intialized by importing ``from imblearn.combine import SMOTETomek`` and resample data using ``.fit_resample(df.values,y)``. Below is the result of oversampling data vs original data.

`smk = SMOTETomek(random_state=42)`<br>
`x_smote, y_smote = smk.fit_resample(df_encode_smote, y)`<br>
`
print(f'Orignal Dataset Shape: {y.shape}')
print(f'SMOTE Dataset Shape: {y_smote.shape}')
print(f"Orignal Dataset Counts {Counter(y)}")
print(f"Resampled Dataset Counts {Counter(y_smote)}")
`

```
Orignal Dataset Shape: (10000,)
SMOTE Dataset Shape: (57890,)
Orignal Dataset Counts Counter({1: 9652, 6: 112, 2: 95, 4: 78, 3: 45, 5: 18})
Resampled Dataset Counts Counter({4: 9652, 5: 9652, 6: 9652, 3: 9648, 2: 9645, 1: 9641})
```

### 🎛️Feature Scaling
Since the features from the dataset have a different range, feature scaling will be perform to improve model convergence, model enhancing, standardize interpretation using ``from sklearn.preprocessing import StandardScaler``,``.fit_transform(feature)``. Below is the feature scaling result.

`def feature_scaling(x):`<br>
    `x = pd.DataFrame(x)
    x_scaled = StandardScaler().fit_transform(x)
    return x_scaled`<br>

`x_scaled = feature_scaling(x)`<br>
`print(x_scaled)`<br>
  
```
[[-1.63475018 -1.40893844 -0.98002969  0.13998718 -0.23744184 -2.05220837]
 [ 0.23069083 -1.40893844 -0.98002969 -0.32179153 -0.03190768 -2.00938944]
 [ 0.23069083 -1.40893844 -1.75703636 -0.03116157  0.17362648 -1.98084349]
 ...
 [ 0.23069083  1.34321758  1.35099031 -0.46387729  0.24213787 -0.51072697]
 [ 0.23069083  0.79278638 -0.20302302 -0.47356495  0.65320618 -1.08164601]
 [ 0.23069083  1.34321758  0.57398364 -0.46064807  0.51618341 -0.62491078]]
```

### 💠Feature Split
The datasets will be splitted into partitions for further model evaluation, fine-tuning, preventing overfitting model and well assessed generalization model. The data will be splitted into:

- Training set (70%)
- Test set (30%)

The feature split can be performed by:<br>
`from sklearn.model_selection import train_test_split`<br> 
`xtrain, xtest, ytrain, ytest = train_test_split(feature, target, test_size=0.3)`

## 🏂Cross Validation and Hyperparameter Tuning
Cross validation with hyperparameter tuning are performed to assess the performance and generalization ability of a model, which performing model training and evaluation iteratively, and then aggregating the results.

The cross validation technique that will be using: Stratified KFolds. For hyperparameter tuning will be using: GridSearchCV, to evaluate the model into k folds and find the best hyperparameters for the classification task. 

K will be set = 5<br>
``skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)``

Model SVM and variations of hyperparameters for cross validation:<br>
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
<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/training%20confusion%20matrix.png?raw=true" width="300" height="280" alt="alt text">

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
<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/test%20confusion%20matrix.png?raw=true" width="300" height="280" alt="alt text">

- All dataset:

```
              precision    recall  f1-score   support

           1       0.97      0.85      0.90      9641
           2       1.00      1.00      1.00      9645
           3       0.96      0.99      0.97      9648
           4       0.99      0.99      0.99      9652
           5       0.91      0.97      0.94      9652
           6       0.97      1.00      0.98      9652

    accuracy                           0.97     57890
   macro avg       0.97      0.97      0.96     57890
weighted avg       0.97      0.97      0.96     57890
```
<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/all%20confusion%20matrix.png?raw=true" width="300" height="280" alt="alt text">

Observation: From the classification report, we can make the following observations for the three different sets: training, test, and all data.

- Precision: Overall the model performs well in terms of precision for all failure types, with values ranging from 0.91 to 1.00. The precision values are consistent across the training, test, and all data sets.

- Recall: Recall values are generally high for all classes, ranging from 0.84 to 1.00. Similar to precision, recall values remain consistent across the different sets.

- F1-Score: F1-scores for all classes are quite high, ranging from 0.90 to 1.00. Again, the F1-scores remain consistent across the different sets.

- Support: Evidence that the class distribution is roughly balanced across all classes in the data sets.

- Accuracy: The overall accuracy of the model is approximately 0.97 for all sets, indicating that the model correctly classifies around 97% of the instances. This high accuracy suggests that the model is performing well on the given data.

- Confusion Matrix: Overall the true positive for all classes has a very high point accross all different sets. This result suggests that the model is performing well in correctly identifying instances of each classes. 


