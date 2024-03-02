from src.features.parser_ml import encoded_dataset, TARGET_COLUMN, dataset
X = encoded_dataset #features used to train the model, remember we don't have the target column here (salary)
y = dataset[TARGET_COLUMN] #target column! this is what we want to predict, used for training

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

# separating data for training and testing the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #20% used for testing, 80% for training

# creating a linear regression instance
linear_regression = LinearRegression()

# training that instance of the linear regression model with our training data
linear_regression.fit(X_train, y_train)

# lets get predictions
y_pred = linear_regression.predict(X_test)

# lets check those predictions performance
print("FIRST PREDICTION -------------------------")
mse = metrics.mean_squared_error(y_test, y_pred) # loss function
print(f"Funcion de perdida: Error cuadrático medio: {mse}")
print(f"Metrica de rendimiento: Raiz del error cuadrático medio: {np.sqrt(mse)}") # performance metric
print(f"Error absoluto medio: {metrics.mean_absolute_error(y_test, y_pred):.0f}")

# something's wrong, there are predictions that are extremely bad (atypical errors):
# print(y_pred[(y_pred > 10e10) | (y_pred < -10e10)])
# probably generated because of overfitting (or underfitting)

# lets clean out those predictions:
valid_examples = (y_pred < 1e10) & (y_pred > -1e10)
print("SECOND PREDICTION -------------------------")
mse = metrics.mean_squared_error(y_test[valid_examples], y_pred[valid_examples])
print(f"Error cuadrático medio: {mse}")
print(f"Raiz del error cuadrático medio: {np.sqrt(mse)}")
print(f"Error absoluto medio: {metrics.mean_absolute_error(y_test[valid_examples], y_pred[valid_examples])}")
# it's still a bad prediction (aprox 100.000 pesos of difference between prediction and real value)

#if model works bad either way, then, overfitting or underfitting is responsible? 