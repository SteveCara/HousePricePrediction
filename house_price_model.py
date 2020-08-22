import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# import the data
home_data = pd.read_csv('train.csv', index_col=0)
test_data = pd.read_csv('test.csv', index_col=0)
test_data_with_ID = pd.read_csv('test.csv')


# set training features X and target y and drop Id in X
y = home_data['SalePrice']
X = home_data.drop(['SalePrice'], axis=1)


# merge train and test data (they will be separated again at the end)**
X = pd.concat([X, test_data], axis=0, sort=False)


# remove features with high proportion of missing values
# EDA revealed a number of fields with >20% missing values
# ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
X.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence',
        'FireplaceQu'], axis=1, inplace=True)


# remove features with mostly single value
# EDA revealed a number of fields with mostly single values
# ['BsmtFinSF2','LowQualFinSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating']
X.drop(['BsmtFinSF2', 'LowQualFinSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
        'PoolArea', 'MiscVal', 'Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating'], axis=1, inplace=True)


# fill missing values
# isolate numerical, categorical and ordinal data (features identified in EDA)
# numerical features
numerical = ["LotFrontage", "GarageArea", "MSZoning", "BsmtHalfBath", "BsmtFullBath", "BsmtFinSF1",
             "BsmtUnfSF", "TotalBsmtSF", "MasVnrArea", 'GarageYrBlt', 'TotRmsAbvGrd', '1stFlrSF', 'GarageCars']
X[numerical] = X[numerical].fillna(X[numerical].mean())
# print(X.isnull().sum().sort_values(ascending=False).head(20))


# ordinal features
ordinal = ['GarageType', 'GarageFinish', 'BsmtFinType2', 'BsmtExposure', 'BsmtFinType1',
           'GarageCond', 'GarageQual', 'BsmtCond', 'BsmtQual', "KitchenQual",
           "HeatingQC", 'ExterQual', 'ExterCond']
X[ordinal] = X[ordinal].fillna("NA")
# print(X.isnull().sum().sort_values(ascending=False).head(10))


# categorical features
categorical = ["MasVnrType", "MSZoning", "Exterior1st",
               "Exterior2nd", "SaleType", "Electrical", "Functional"]
X[categorical] = X[categorical].transform(lambda x: x.fillna(x.mode()[0]))
# print(X.isnull().sum().sort_values(ascending=False).head(5))


# Ordinal mapping
# EDA revealed there are three datatypes for categorical data  1.quality, 2.finished, 3.expose
quality_col = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
               'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']
quality = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA': 0}
for q in quality_col:
    X[q] = X[q].map(quality)

finished_col = ['BsmtFinType1', 'BsmtFinType2']
finished = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4,
            'Rec': 3, 'LwQ': 2, 'Unf': 1, 'NA': 0}
for f in finished_col:
    X[f] = X[f].map(finished)

expose_col = ['BsmtExposure']
expose = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
for e in expose_col:
    X[e] = X[e].map(expose)


# get dummy data
X = pd.get_dummies(X, drop_first=True)


# consolidation of data
X_new = X.loc[home_data.index]
test = X.loc[test_data.index]


# train test split
X_train, X_valid, y_train, y_valid = train_test_split(
    X_new, y, test_size=0.2, random_state=42)


# Model XGBoost Tuned with RandomizedSearchCV
parameters = {"learning_rate": [0.1, 0.01, 0.001],
              "gamma": [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2],
              "max_depth": [2, 4, 6, 8, 10],
              "colsample_bytree": [0.2, 0.4, 0.6, 0.8, 1.0],
              "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],
              "reg_alpha": [0, 0.5, 1],
              "reg_lambda": [1, 1.5, 2, 3, 4.5],
              "min_child_weight": [1, 3, 5, 7],
              "n_estimators": [100, 250, 500, 1000]}


xgb_gs = RandomizedSearchCV(XGBRegressor(), param_distributions=parameters,
                            scoring="neg_mean_absolute_error", cv=3)

xgb_gs.fit(X_train, y_train)

# make prediction & check mae
pred_1 = xgb_gs.best_estimator_.predict(X_valid)
print('mae xgb', mean_absolute_error(y_valid, pred_1))


# Model RandomForest Tuned with RandomizedSearchCV
parameters_rf = {'n_estimators': [100, 250, 500, 1000], 'criterion': (
    'mse', 'mae'), 'max_depth': [2, 4, 6, 8, 10], 'max_features': ('auto', 'sqrt', 'log2')}


rf_gs = RandomizedSearchCV(RandomForestRegressor(), param_distributions=parameters_rf,
                           scoring="neg_mean_absolute_error", cv=3)

rf_gs.fit(X_train, y_train)

# make prediction & check mae
pred_2 = rf_gs.best_estimator_.predict(X_valid)
print('mae rf', mean_absolute_error(y_valid, pred_2))


# make prediction for submission
pred_submission = xgb_gs.best_estimator_.predict(test)


# Format for results submission
output = pd.DataFrame({'Id': test_data_with_ID.Id,
                       'SalePrice': pred_submission})
output.to_csv('submission_3.csv', index=False)
