# House Price Prediction

- Created a tool which predicts house prices in Iowa, US

- Extracted, transformed and loaded the Ames Iowa house price data

- Completed some explaratory data analysis to work out which features would be best for training the models

- Undertook feature engineering to improve the data's compatibility with the models

- Created and trained Random Forest Regression and Extreme Gradient Boosted (XGBoost) models

- Optimised models using RandomizedSearchCV

- Created prediction outputs to submit to the Kaggle.com House Prices competition (currenlty in the top 7% of submissions as of 22.08.2020)

## Resources

**Python Version:** 3.7

**Packages:** pandas, numpy, matplotlib, seaborn, sklearn, xgboost

**Competition:** https://www.kaggle.com/c/home-data-for-ml-course

**Project Reference 1:** https://www.kaggle.com/kshivi99/housing-prices-from-eda-to-submission-top-6/notebook#12.-Training

**Project Reference 2:** https://www.kaggle.com/angqx95/data-science-workflow-top-2-with-tuning/notebook

Links to kernels and blogs I used when learning to complete this analyis. Thanks to them!


## EDA
I completed some exploratory data analysis to better understand the data. It was clear that there were a number of missing ('NaN') values in the data and that some feature engineering would be required before training the model. I also investigated the correlations between some of the key metrics and the target feature 'SalePrice'. Some highlights are below.


