import numpy as np
from sklearn.linear_model import LinearRegression

import logging

# Initialise Logger
logging.basicConfig(level=logging.INFO, format="[{asctime}] - {funcName} - {message}", style='{')
logger = logging.getLogger("LINEAR_REGRESSION")

def main()-> None:
    pass

if __name__ == '__main__':
    try:
        logger.info("Starting Linear Regression course")

        main()

        logger.info("Finished Linear Regression course")
    except Exception as e:
        logger.error(f"{str(e)}")
