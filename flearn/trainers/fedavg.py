import os

import numpy as np
import sys
from .grpc import run_server, stop_server
from typing import Dict
import utils
import flwr as fl

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Tuple, Union, List
import openml

def fit_round(rnd: int) -> Dict:
    # Send round to client
    return {"rnd", rnd} 

def get_eval_fn(model: LogisticRegression):
    # get_eval_fn
    _, (X_test, y_test) = utils.load_mnist()

    # The `evaluate` function will be called after every round
    def evaluate(parameters: fl.common.Weights):
        # Update model with the latest parameters
        set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy * 10}

    return evaluate
 
def get_strategy(number_clients, model):
    # get strategy
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=number_clients,
        eval_fn=get_eval_fn(model),
        on_fit_config_fn=fit_round,
    )
    return strategy

def Server (min_num_clients, rounds, server_address):
    print('------------Using Federated avg to Train------------')

    model = LogisticRegression()
    utils.set_initial_params(model)

    strategy = get_strategy(min_num_clients, model)

    client_manager = fl.server.SimpleClientManager()
    server = fl.server.Server(client_manager=client_manager, strategy=strategy)
    # Run server
    grpc_server = run_server(client_manager, server_address)

    hist = server.fit(num_rounds=rounds)
    print(hist)
    # Stop server
    stop_server(grpc_server)