# ⚙️🧬Failure Machine Prediction
A classification model that predicts a machine failure types. Possible machine failures:

- No Failure
- Heat Dissipation Failure
- Power Failure
- Overstain Failure
- Tool Wear Failure
- Random Failures

The model are built on Support Vector Machine (SVM). Inbalanced dataset solved using Synthetic Minority Oversampling Technique (SMOTE). Cross validation using Stratified Kfolds, SearchGridCV. Model evaluation using Precision/Recall, F1 Score, Confusion Matrix, and Accuracy.

<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/training%20confusion%20matrix.png?raw=true" alt="alt text" width="350" height="270"> <img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/images/test%20confusion%20matrix.png?raw=true" alt="alt text" width="350" height="270"> 

Model features:
- Type: L, M, H
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

## Data Acquisition
Dataset can be obtained from Kaggle: https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification
