# Importing Essential libraries
import pandas as pd
import pickle

# Loading the dataset
df = pd.read_csv('ipl.csv')

pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
print("Dataset head :\n", df.head())
print(df.isnull().sum())

# --- Data Cleaning ---
# Removing unwanted columns
columns_to_remove = ['mid', 'venue', 'batsman', 'bowler', 'striker', 'non-striker']
df.drop(labels=columns_to_remove, axis=1, inplace=True)

print("Total teams :\n", df['bat_team'].unique())

# Keeping only consistent teams
consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Daredevils', 'Sunrisers Hyderabad']

df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]

# Removing the first 5 overs data in every match
df = df[df['overs'] >= 5.0]

print(df.head())
print("Bat_team :\n", df['bat_team'].unique())
print("Bowl_team :\n", df['bowl_team'].unique())

# Converting the column 'date' from string into datetime object
from datetime import datetime

df['date'] = df['date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
print(df.head())

# --- Data Preprocessing ---
# Converting categorical features using OneHotEncoding method
encoded_df = pd.get_dummies(data=df, columns=['bat_team', 'bowl_team'])
print("Converted categorical features using OneHotEncoding method :\n", encoded_df.head())

# Rearranging the columns
encoded_df = encoded_df[
    ['date', 'bat_team_Chennai Super Kings', 'bat_team_Delhi Daredevils', 'bat_team_Kings XI Punjab',
     'bat_team_Kolkata Knight Riders', 'bat_team_Mumbai Indians', 'bat_team_Rajasthan Royals',
     'bat_team_Royal Challengers Bangalore', 'bat_team_Sunrisers Hyderabad',
     'bowl_team_Chennai Super Kings', 'bowl_team_Delhi Daredevils', 'bowl_team_Kings XI Punjab',
     'bowl_team_Kolkata Knight Riders', 'bowl_team_Mumbai Indians', 'bowl_team_Rajasthan Royals',
     'bowl_team_Royal Challengers Bangalore', 'bowl_team_Sunrisers Hyderabad',
     'overs', 'runs', 'wickets', 'runs_last_5', 'wickets_last_5', 'total']]

# Splitting the data into train and test set
X_train = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year <= 2016]
X_test = encoded_df.drop(labels='total', axis=1)[encoded_df['date'].dt.year >= 2017]

y_train = encoded_df[encoded_df['date'].dt.year <= 2016]['total'].values
y_test = encoded_df[encoded_df['date'].dt.year >= 2017]['total'].values

# Removing the 'date' column
X_train.drop(labels='date', axis=True, inplace=True)
X_test.drop(labels='date', axis=True, inplace=True)

# --- Model Building ---
# Linear Regression Model
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)
prediction = regressor.predict(X_test)

from sklearn import metrics
from sklearn.metrics import r2_score
import numpy as np

print('LR_MAE:', metrics.mean_absolute_error(y_test, prediction))
print('LR_MSE:', metrics.mean_squared_error(y_test, prediction))
print('LR_RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
print("LR_r2_score" , r2_score(y_test, prediction))

# Creating a pickle file for the classifier
# filename = 'first-innings-score-lr-model.pkl'
# pickle.dump(regressor, open(filename, 'wb'))

# Ridge Regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

ridge = Ridge()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40]}
ridge_regressor = GridSearchCV(ridge, parameters, scoring='neg_mean_squared_error', cv=5)
ridge_regressor.fit(X_train, y_train)

GridSearchCV(cv=5, estimator=Ridge(),
             param_grid={'alpha': [1e-15, 1e-10, 1e-08, 0.001, 0.01, 1, 5, 10,
                                   20, 30, 35, 40]},
             scoring='neg_mean_squared_error')

print(ridge_regressor.best_params_)
print(ridge_regressor.best_score_)

prediction = ridge_regressor.predict(X_test)

import matplotlib.pyplot as plt
import seaborn as sns

sns.distplot(y_test - prediction)
plt.show()

print('RR_MAE:', metrics.mean_absolute_error(y_test, prediction))
print('RR_MSE:', metrics.mean_squared_error(y_test, prediction))
print('RR_RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
print("RR_r2_score" , r2_score(y_test, prediction))

# Creating a pickle file for the classifier
filename = 'first-innings-score-lr-model.pkl'
pickle.dump(regressor, open(filename, 'wb'))

# Lasso Regression

from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

lasso = Lasso()
parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40]}
lasso_regressor = GridSearchCV(lasso, parameters, scoring='neg_mean_squared_error', cv=5)

lasso_regressor.fit(X_train, y_train)
print(lasso_regressor.best_params_)
print(lasso_regressor.best_score_)

prediction = lasso_regressor.predict(X_test)
sns.distplot(y_test - prediction)
plt.show()

print('LR_MAE:', metrics.mean_absolute_error(y_test, prediction))
print('LR_MSE:', metrics.mean_squared_error(y_test, prediction))
print('LR_RMSE:', np.sqrt(metrics.mean_squared_error(y_test, prediction)))
print("LR_r2_score" , r2_score(y_test, prediction))

