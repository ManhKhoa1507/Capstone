import flwr as fl

from flwr.common import GRPC_MAX_MESSAGE_LENGTH
from flwr.common.logger import log
from flwr.server.grpc_server.grpc_server import start_grpc_server

client_manager = fl.server.SimpleClientManager()
server = fl.server.Server(client_manager=client_manager)

def run_server(client_manager, server_address):
    # Run server
    print(f"\nStarting GRPC server at {server_address}\n")
    grpc_server = start_grpc_server(
        client_manager=server.client_manager(),
        server_address=server_address,
        max_message_length=GRPC_MAX_MESSAGE_LENGTH,
    )
    return grpc_server

def stop_server(grpc_server):
    # Stop GRPC server
    print("Stop GRPC server")
    grpc_server.stop(1)
    return