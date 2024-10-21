import pandas as pd
import matplotlib.pyplot as plt
import pickle
import warnings
import seaborn as sns

df = pd.read_csv(r"C:\Users\jayas\Downloads\BTC-USD.csv")

df['year'] = (pd.to_datetime(df['Date']).dt.year)

df.drop('Date',axis = 1,inplace = True)

max_amount = max(df['High'])
max_amount_year = df[df['High'] == max_amount]['year'].values[0]

plt.figure(figsize=[10,7])
sns.lineplot(data = df,x = 'year',y = 'Volume')
sns.stripplot(data = df,x = 'year',y = 'Adj Close',color = 'red')
plt.title('Growth of the Bitcoin by Each Year')
sns.barplot(data = df,x = 'year',y = 'Volume',color = 'blue',ci = None)
sns.boxplot(data = df,x = 'year')

y = df['Adj Close']
x = df['year']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 0,test_size = 0.20)
x_train = x_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.metrics import mean_squared_error,r2_score
from sklearn.tree import DecisionTreeRegressor
lnr = DecisionTreeRegressor()
lnr.fit(x_train,y_train)
ypred1 = lnr.predict(x_test)
r21 = r2_score(ypred1,y_test)
acc1 = mean_squared_error(ypred1,y_test)

from sklearn.ensemble import RandomForestRegressor
rnd = RandomForestRegressor()
rnd.fit(x_train,y_train)
ypred2 = rnd.predict(x_test)
r22 = r2_score(ypred2,y_test)
acc2 = mean_squared_error(ypred2,y_test)

import lightgbm as lgb
lgb = lgb.LGBMRegressor()
lgb.fit(x_train,y_train)
ypred4 = lgb.predict(x_test)
r23 = r2_score(ypred4,y_test)
acc4 = mean_squared_error(ypred4,y_test)

from sklearn.model_selection import GridSearchCV
param_grid = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 1000],
    'max_depth': [-1, 10, 20]
}

grid_search = GridSearchCV(estimator=lgb, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
grid_search.fit(x_train, y_train)
best_model = grid_search.best_estimator_
ypred4 = best_model.predict(x_test)
r26 = r2_score(y_test, ypred4)
acc6 = mean_squared_error(y_test, ypred4)

import xgboost as xgb
xgb = xgb.XGBRegressor()
xgb.fit(x_train,y_train)
ypred5 = xgb.predict(x_test)
r24 = r2_score(ypred5,y_test)
acc5 = mean_squared_error(ypred5,y_test)

from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(x_train,y_train)
ypred6 = gbr.predict(x_test)
r25 = r2_score(ypred6,y_test)
acc7 = mean_squared_error(ypred6,y_test)

print(f"Decision Tree Regressor (Tuned) - MSE: {acc1}, R2: {r21}")
print(f"Random Forest Regressor (Tuned) - MSE: {acc2}, R2: {r22}")
print(f"Gradient Boosting Regressor - MSE: {acc7}, R2: {r25}")
print(f"LightGBM Regressor (Tuned) - MSE: {acc4}, R2: {r23}")
print(f"XGBoost Regressor (Tuned) - MSE: {acc5}, R2: {r24}")

with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rnd, file)

with open('random_forest_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    y_pred_loaded = loaded_model.predict(x_test)