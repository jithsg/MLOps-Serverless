from typing import Dict, Union, Optional, Callable
import os

import pandas as pd
from comet_ml import Experiment
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso


from preprocessing import transform_ts_data_into_features_and_target
from logger import get_console_logger
logger = get_console_logger()
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Now you can safely use the environment variable
api_key = os.environ.get("COMET_ML_API_KEY")
workspace =os.environ.get("COMET_ML_WORKSPACE")


def get_baseline_model_error(X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Returns the baseline model error."""
    predictions = X_test['price_1_hour_ago']
    return mean_absolute_error(y_test, predictions)


def train(
    X: pd.DataFrame,
    y: pd.Series,
    ) -> None:
    """
    Train a boosting tree model using the input features `X` and targets `y`,
    possibly running hyperparameter tuning.
    """
    experiment = Experiment(
        api_key=api_key, 
        workspace=workspace,
        project_name = "pao"
    )
    experiment.add_tag('baseline_model')

    # split the data into train and test
    train_sample_size = int(0.9 * len(X))
    X_train, X_test = X[:train_sample_size], X[train_sample_size:]
    y_train, y_test = y[:train_sample_size], y[train_sample_size:]
    logger.info(f'Train sample size: {len(X_train)}')
    logger.info(f'Test sample size: {len(X_test)}')

    # baseline model performance
    baseline_mae = get_baseline_model_error(X_test, y_test)
    logger.info(f'Test MAE: {baseline_mae}')
    experiment.log_metrics({'Test_MAE': baseline_mae})


if __name__ == '__main__':

    logger.info('Generating features and targets')
    features, target = transform_ts_data_into_features_and_target()
    
    logger.info('Starting training')
    train(features, target)