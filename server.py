import flwr as fl

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.server.grpc_server.grpc_server import start_grpc_server

import importlib

import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict

import argparse

DEFAULT_SERVER_ADDRESS = "0.0.0.0:9000"


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


def get_argument():
    # Get the argument from terminal
    parser = argparse.ArgumentParser(description="Flower")

    parser.add_argument(
        "--optimizer",
        type=str,
        default="fedavg",
        help=f"choose between fedavg and fed+ (default: fedavg)",
    )

    parser.add_argument(
        "--server_address",
        type=str,
        default=DEFAULT_SERVER_ADDRESS,
        help=f"gRPC server address (default: {DEFAULT_SERVER_ADDRESS})",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of rounds (default: 1)",
    )

    parser.add_argument(
        "--min_num_clients",
        type=int,
        default=2,
        help="Minimum number of available clients required for sampling (default: 2)",
    )

    args = parser.parse_args()
    dict_args = vars(args)

    return args


def run_server(client_manager, server_address):
    # Run server
    grpc_server = start_grpc_server(
        client_manager=server.client_manager(),
        server_address=DEFAULT_SERVER_ADDRESS,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
    )
    return grpc_server


if __name__ == "__main__":
    args = get_argument()

    # model = LogisticRegression()
    # utils.set_initial_params(model)

    # strategy = get_strategy(args.min_num_clients, model)

    client_manager = fl.server.SimpleClientManager()
    # server = fl.server.Server(client_manager=client_manager, strategy=strategy)
    
    opt_path='flearn.trainers.%s' % parser['optimizer']
    mod = importlib.import_module(opt_path)
    optimizer = getattr(mod, 'Server')

    # Run server
    grpc_server = run_server(client_manager, args.server_address)

    # # Fit model
    # hist = server.fit(num_rounds=args.rounds)

    # Stop server
    grpc_server.stop(1)
