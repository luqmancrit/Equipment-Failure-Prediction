# ⚙️🧬Failure Machine Prediction
A prediction model that predicts a machine failure. Possible machine failures:

- No Failure
- Heat Dissipation Failure
- Power Failure
- Overstain Failure
- Tool Wear Failure
- Random Failures

Features:
- Type: L, M, H


The model are built on Support Vector Machine. Inbalanced dataset solved using Synthetic Minority Oversampling Technique (SMOTE). Cross validation using Stratified Kfolds, SearchGridCV. Model evaluation using precision/recall, f1 score, confusion matrix, and accuracy.

<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/training%20confusion%20matrix.png?raw=true" alt="alt text" width="350" height="270"> <img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/test%20confusion%20matrix.png?raw=true" alt="alt text" width="350" height="270"> 

<img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/classification%20report%20-%20training.png?raw=true" alt="alt text" width="350" height="200"> <img src="https://github.com/luqmancrit/Failure-Prediction/blob/main/classification%20report%20-%20test.png?raw=true" alt="alt text" width="350" height="200">
