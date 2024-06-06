import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge

data = pd.read_csv('C:/Users/mohamed/Downloads/Automobile_data.csv')

data.replace('?', np.nan, inplace=True)

numeric_cols = data.select_dtypes(include=['object']).columns
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='ignore')

data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype.kind in 'biufc' else x)

data = data.apply(lambda x: x.fillna(x.mode()[0]) if x.dtype.kind not in 'biufc' else x)

X = data.drop('price', axis=1)

X = pd.get_dummies(X, drop_first=True)

y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'alpha': [0.1, 1.0, 10.0, 100.0, 200.0]
}

ridge = Ridge()
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best Parameters: {best_params}')
print(f'Best Cross-Validation R^2 Score: {best_score}')

# Evaluate the best model on the test set
best_ridge = grid_search.best_estimator_
y_pred_ridge = best_ridge.predict(X_test)

mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f'Test Set Mean Squared Error: {mse_ridge}')
print(f'Test Set R^2 Score: {r2_ridge}')