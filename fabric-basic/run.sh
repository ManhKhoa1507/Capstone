#!/bin/sh

echo 'STARTED NETWORK BOOTSTRAPPING'

export IMAGE_TAG=2.1.0
export CA_TAG=latest
export COMPOSE_PROJECT_NAME=base-network

# generate initial configurations
bash init.sh

export CA_LEFT=$(cd crypto-config/peerOrganizations/base.left/ca && ls *_sk)

export CA_RIGHT=$(cd crypto-config/peerOrganizations/base.right/ca && ls *_sk)


docker-compose -f  docker-compose.yaml up -d

echo '---LOGIN TO CLI CONTAINER----'

docker exec  cli  bash -c "scripts/script.sh"

echo 'COMPLETED'