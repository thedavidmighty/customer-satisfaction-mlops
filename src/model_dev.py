import logging
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

from typing import Union
from sklearn.linear_model import LinearRegression

class Model(ABC):
    """
    Abstract class for all Models
    """

    @abstractmethod
    def train(self, X_train, y_train):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training label

        Returns:
            None   
        """
        pass
        # try:


        # except Exception as e:
        #     logging.error("Error training model: {}".format(e))
        #     raise e    

class LinearRegressionModel(Model):
    """
    Linear regresion model
    """
    def train(self, X_train, y_train, **kwargs):
        """
        Trains the model
        Args:
            X_train: Training data
            y_train: Training label

        Returns:
            None   
        """
        try: 
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            logging.info("Model training complete")
            return reg
        
        except Exception as e:
            logging.error("Error running Model: {}".format(e))
            raise e
