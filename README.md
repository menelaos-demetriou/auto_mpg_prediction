# auto_mpg_prediction
Dataset was taken from https://archive.ics.uci.edu/ml/datasets/Auto+MPG  
This repo contains the implementation of an estimator for auto mpg prediction.  
So far SGDRegresson, GradientBoostingRegressor, RandomForestRegressor and SVR are used.  
Best candidate:{'regressor': SVR(C=1000, gamma=0.001), 'regressor__C': 1000, 'regressor__gamma': 0.001, 'regressor__kernel': 'rbf'}  
Evaluation on test set: Root Mean squared error is 3.21

# Further improvements:
ETL for car names into company name.  
Feature selection.  

# Requirements
pandas  
numpy  
scikit-learn  
seaborn  
matplotlib  


