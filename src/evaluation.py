import logging
from tkinter import _test
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from abc import ABC, abstractmethod

class Evaluation(ABC):
    """
    Abstract calss defining strategy for evaluation our models
    """

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.array):
        """
        Calculates the scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
            Returns:
                None
        """
class MSE(Evaluation):
    """
    Evaluation Strategy that uses MEan Squared error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.array):
        """
        Calculates the MSE for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
            Returns:
                None
        """
        try:
            logging.info("Calculating MSE")
            mse = mean_squared_error(y_true, y_pred)
            logging.info("MSE: {}".format(mse))
            return mse
        
        except Exception as e:
            logging.error("Error in calculating MSE: {}".format(e))
            raise e

class R2(Evaluation):
    """
    Evaluation Strategy that uses MEan Squared error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.array):
        """
        Calculates the R2-scores for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
            Returns:
                None
        """
        try:
            logging.info("Calculating R2 Score")
            r2 = r2_score(y_true, y_pred)
            logging.info("r2_score:{}".format(r2))
            return r2
        except Exception as e:
            logging.error("Error in calculating R2-Score: {}".format(e))

class RMSE(Evaluation):
    """
    Evaluation Strategy that uses Mean Squared error
    """
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.array):
        """
        Calculates the RMSE for the model
        Args:
            y_true: True labels
            y_pred: Predicted labels
        
            Returns:
                None
        """
        try:
            logging.info("Calculating RMSE")
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e