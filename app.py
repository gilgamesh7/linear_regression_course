import numpy as np
from sklearn.linear_model import LinearRegression
# For polynomial regression
from sklearn.preprocessing import PolynomialFeatures
# For advanced inear Regression With statsmodels
import statsmodels.api as sm

import logging

# Initialise Logger
logging.basicConfig(level=logging.INFO, format="[{asctime}] - {funcName} - {message}", style='{')
logger = logging.getLogger("LINEAR_REGRESSION")

def simple_linear_regression()-> None:
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
        logger.info(f"predicted response using predict():\n{y_prediction}")

        # Predict Response using Matrix Multiplication
        y_prediction_2 = model.intercept_ + model.coef_ * x
        logger.info(f"predicted response using b0+b1*x:\n{y_prediction_2.reshape(-1)}")


    except Exception as e:
        logger.error(f"{str(e)}")


def multiple_linear_regression()-> None:
    try:
        logger.info("Setting up regressors & response")

        # In multiple linear regression, x is a multi-dimensional array with at least two columns, while y is usually a one-dimensional array.
        x = [
            [0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]
        ]
        y = [4, 5, 20, 14, 32, 22, 38, 43]

        x, y = np.array(x), np.array(y)

        logger.info("Create & Fit Multiple Linear Regression model")
        #  calculate the optimal values of the weights 𝑏₀ and 𝑏₁, using the existing input and output, x and y, as the arguments.
        model = LinearRegression().fit(x, y)

        r_sq = model.score(x, y)
        logger.info(f"coefficient of determination: {r_sq}")
        logger.info(f"intercept: {model.intercept_}")
        logger.info(f"coefficients: {model.coef_}")

        # Predictions
        y_pred = model.predict(x)
        logger.info(f"predicted response:\n{y_pred}")

        x_new = np.arange(10).reshape((-1, 2))
        y_new = model.predict(x_new)
        logger.info(f"Prediction for new data set of x response:\n{y_new}")

    except Exception as e:
        logger.error(f"{str(e)}")

def polynomial_regression()-> None:
    try:
        logger.info("Setting up regressors & response")

        # you need to transform the array of inputs to include nonlinear terms such as 𝑥².
        # you need the input to be a two-dimensional array. That’s why .reshape() is used.
        x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
        y = np.array([15, 11, 2, 8, 25, 32])

        # transformer refers to an instance of PolynomialFeatures that you can use to transform the input x.
        transformer = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)

        logger.info("Fit polynomial to transformer")
        transformer.fit(x)

        logger.info("input array as the argument and returns the modified array")
        x_ = transformer.transform(x)

        # modified input array contains two columns: one with the original inputs and the other with their squares
        logger.info(f"_x : \n {x_}")

        logger.info(f"Create & Fit model")
        model = LinearRegression().fit(x_, y)

        r_sq = model.score(x_, y)
        logger.info(f"coefficient of determination: {r_sq}")
        logger.info(f"intercept: {model.intercept_}")
        logger.info(f"coefficients: {model.coef_}")

        logger.info(f"Predict response")
        y_pred = model.predict(x_)
        print(f"predicted response:\n{y_pred}")

    except Exception as e:
        logger.error(f"{str(e)}")

def advanced_regression_with_statsmodels()-> None:
    try:
        logger.info("Setting up regressors & response")
        x = [
            [0, 1], [5, 1], [15, 2], [25, 5], [35, 11], [45, 15], [55, 34], [60, 35]
        ]
        y = [4, 5, 20, 14, 32, 22, 38, 43]
        x, y = np.array(x), np.array(y)

        #  add the column of ones to the inputs if you want statsmodels to calculate the intercept 𝑏₀.
        x = sm.add_constant(x)

        logger.info(f"Regressors & reponse : \n {x = } \n {y = }")

        logger.info(f"Create model & Fit")
        # Notice that the first argument is the output, followed by the input
        model = sm.OLS(y, x)
        results = model.fit()

        logger.info(f"Results \n : {results.summary()}")
        logger.info(f"coefficient of determination: {results.rsquared}")
        logger.info(f"adjusted coefficient of determination: {results.rsquared_adj}")
        logger.info(f"regression coefficients: {results.params}")

        logger.info(f"predicted response:\n{results.fittedvalues}")
        logger.info(f"predicted response:\n{results.predict(x)}")

        x_new = sm.add_constant(np.arange(10).reshape((-1, 2)))
        logger.info(f"New regressor : \n {x_new}")
        logger.info(f"New prediction : \n {results.predict(x_new)}")




    except Exception as e:
        logger.error(f"{str(e)}")

def main()-> None:
    try:
        logger.info("Simple Linear Regression With scikit-learn")
        simple_linear_regression()

        logger.info("Multiple Linear Regression With scikit-learn")
        multiple_linear_regression()

        logger.info("Polynomial Regression with scikit-learn")
        polynomial_regression()

        logger.info(f"Adnavced linear regression with statsmodels")
        advanced_regression_with_statsmodels()

    except Exception as e:
        logger.error(f"{str(e)}")

if __name__ == '__main__':
    try:
        logger.info("Starting Linear Regression course")

        main()

        logger.info("Finished Linear Regression course")
    except Exception as e:
        logger.error(f"{str(e)}")
