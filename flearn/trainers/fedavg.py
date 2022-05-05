import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf
import sys
from typing import Dict

import flwr as fl

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.server.grpc_server.grpc_server import start_grpc_server

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from fedbase import BaseFedarated

def process_grad(grads):
        # a flattened grad in numpy (1-D array)
    client_grads = grads[0]

    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i]) # output a flattened array


    return client_grads

class Server(BaseFedarated):
    def __init__(self, model, min_num_clients, rounds):
        print('Using Federated avg to Train')
        model = LogisticRegression()
        set_initial_params(model)

        strategy = get_strategy(min_num_clients, model)

        client_manager = fl.server.SimpleClientManager()
        server = fl.server.Server(client_manager=client_manager, strategy=strategy)

    def fit_round(rnd: int) -> Dict:
        # Send round to client
        return {"rnd", rnd}

    def get_eval_fn(model: LogisticRegression):
        # get_eval_fn
        _, (X_test, y_test) = utils.load_mnist()

        # The `evaluate` function will be called after every round
        def evaluate(parameters: fl.common.Weights):
            # Update model with the latest parameters
            utils.set_model_params(model, parameters)
            loss = log_loss(y_test, model.predict_proba(X_test))
            accuracy = model.score(X_test, y_test)
            return loss, {"accuracy": accuracy}

        return evaluate

    def get_strategy(number_client, model):
        # get strategy
        strategy = fl.server.strategy.FedAvg(
            min_available_clients=number_client,
            eval_fn=get_eval_fn(model),
            on_fit_config_fn=fit_round,
        )
        return strategy
    
    def set_initial_params(model: LogisticRegression):
        n_classes = 10  # MNIST has 10 classes
        n_features = 784  # Number of features in dataset
        model.classes_ = np.array([i for i in range(10)])

        model.coef_ = np.zeros((n_classes, n_features))
        if model.fit_intercept:
            model.intercept_ = np.zeros((n_classes,))