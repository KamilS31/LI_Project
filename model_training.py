from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Function to train the model
def train_models(X_train, y_train):
    model = RandomForestRegressor()
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_absolute_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    return best_model

# Function to evaluate the model
def evaluate_model(predictions, true_values):
    mae = mean_absolute_error(true_values, predictions)
    mse = mean_squared_error(true_values, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(true_values, predictions)


    print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')
