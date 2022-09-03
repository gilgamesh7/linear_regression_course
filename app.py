import numpy as np
from sklearn.linear_model import LinearRegression

import logging

# Initialise Logger
logging.basicConfig(level=logging.INFO, format="[{asctime}] - {funcName} - {message}", style='{')
logger = logging.getLogger("LINEAR_REGRESSION")

def simple_linear_regression():
    try:
        logger.info("Setting up regressors & response")

        # The inputs (regressors, 𝑥) and output (response, 𝑦) should be arrays or similar objects
        # call .reshape() on x because this array must be two-dimensional, or more precisely, it must have one column and as many rows as necessary.
        x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
        y = np.array([5, 20, 14, 32, 22, 38])

        # Create model from regressors & response and fit
        logger.info("Create Linear Regression model")
        model = LinearRegression(fit_intercept=True, normalize=False, copy_X =True, n_jobs=1)

        logger.info("Fit Linear Regression model")
        #  calculate the optimal values of the weights 𝑏₀ and 𝑏₁, using the existing input and output, x and y, as the arguments.
        model.fit(x, y)

        logger.info("Get co-efficient of determination R^2")
        coefficient_of_determination = model.score(x,y)

        logger.info(f"\n Coefficient of determination (R^2) : {coefficient_of_determination} \n Intercept (b0) : {model.intercept_} \n slope (b1): {model.coef_}")

        # Predict Response
        y_prediction = model.predict(x)
        print(f"predicted response using predict():\n{y_prediction}")

        # Predict Response using Matrix Multiplication
        y_prediction_2 = model.intercept_ + model.coef_ * x
        print(f"predicted response using b0+b1*x:\n{y_prediction_2.reshape(-1)}")


    except Exception as e:
        logger.error(f"{str(e)}")

def main()-> None:
    try:
        logger.info("Simple Linear Regression With scikit-learn")
        simple_linear_regression()
    except Exception as e:
        logger.error(f"{str(e)}")

if __name__ == '__main__':
    try:
        logger.info("Starting Linear Regression course")

        main()

        logger.info("Finished Linear Regression course")
    except Exception as e:
        logger.error(f"{str(e)}")
