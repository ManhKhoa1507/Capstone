import flwr as fl

import importlib

import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict

import argparse

DEFAULT_SERVER_ADDRESS = "0.0.0.0:9000"


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
        help="Number of rounds (default: 5)",
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


if __name__ == "__main__":
    args = get_argument()

    client_manager = fl.server.SimpleClientManager()
    server = fl.server.Server(client_manager=client_manager)

    opt_path = 'flearn.trainers.'
    if (args.optimizer == 'fedavg'):
        opt_path += 'fedavg'

        mod = importlib.import_module(opt_path)
        optimizer = mod.Server(args.min_num_clients,
                               args.rounds, args.server_address)
