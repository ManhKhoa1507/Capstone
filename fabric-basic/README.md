
**HLF**

Tutorial on deploy a simple HLF network using docker-compose 

HLF version: 2.1.0

**Network Details**

This network has the following configurations

  1. Three ordering node using Raft consensus

  2. Two organizations will have one peer node each

  3. Two certificate authorites for the each organizations

 On the successfull network creation we will have 8 containers in total 3 for the ordering node ,2 for the peer nodes,2 for the CA and one for the fabric cli tools

 We are going to use the `cli` container for the channel creation,peer joining and the peer updation operations 

Note:In this tutorial we are using `cryptogen` for generating the certs but its not recommended for the production networks.For the production network use Fabric CA or anyother CA


**Project directory**

 Manily we have four files in the directory 

 1.`base.yaml` we are extending this into our `docker-compose.yaml` it includes common configurations for order,peer and cli containers.

 2.`docker-compose.yaml` Main file for the network creation it includes docker container configuration for the whole network

 3. `crypto-config.yaml` contains  configurations for the certificate generation

 4. `configtx.yaml` contains the policy's for orgz,configurations for the order node and channel


Apart from  the above i've added  bash scripts for automating  of the network operations

1. `run.sh`-Bootstrap the netowrk with channel create,peer joining and peer updates.

3. `down.sh` For  down the network and delete and do a force delete  


**Prerequisites**

1. Make sure your environment is satisfied with  https://hyperledger-fabric.readthedocs.io/en/release-1.1/prereqs.html

2. Install `cryptogen`  and `configtxgen` for genearting cerficates and channel artifcats https://stackoverflow.com/questions/45498921/steps-to-install-cryptogen-tool-for-hyperledger-fabric-node-setup

 
 **Steps**

 Open terminal in the  project root directory and do the following steps.

 1. Set up an environment variable for all operations `export FABRIC_CFG_PATH=${PWD}`

 2. Generate the certifcates using 
    ```
    cryptogen generate --config=./crypto-config.yaml
    ```

3. Generate the gensis block for the order nodes
    ```
    configtxgen -profile Raft  -channelID base-sys-channel -outputBlock ./artifacts/genesis.block
    ```
This will generate a file named `gensis.block` in the `artifacts` directory. 

Now we are going to create the channel.Since this is our main channel, we name it as `basechanel` itself.

`export CHANNEL_NAME=basechannel`

4. Generate the channel artifacts 
    ```
    configtxgen -profile MainChannel -outputCreateChannelTx  ./artifacts/channel.tx -channelID $CHANNEL_NAME
    ```

5. Generate the anchor peer updates for the org1

    ```
    configtxgen -profile MainChannel -outputAnchorPeersUpdate  ./artifacts/BaseLeftOrg.tx -channelID $CHANNEL_NAME -asOrg LeftOrgMSP
    ```

6. Generate the anchor peer update for org2

      ```
      configtxgen -profile MainChannel -outputAnchorPeersUpdate  ./artifacts/BaseRightOrg.tx -channelID $CHANNEL_NAME -asOrg RightOrgMSP
      ```

For running CA for each organization we have  to change the value of `FABRIC_CA_SERVER_TLS_KEYFILE` variable in `docker-compose.yaml`

We will show this for the first organization repeat the same steps(7 and 8) for other organization as well.

 7. Go to the folder `crypto-config/peerOrganizations/base.left/ca`

 8. In that folder we can see file name with long hash as name and end with `_sk`, copy this file name and replace and update the `FABRIC_CA_SERVER_TLS_KEYFILE` under the `ca_left` container configuration in `docker-compose.yaml` file also update this value in command in command section after `--ca` keyfile flag 

eg:

```
command: sh -c 'fabric-ca-server start --ca.certfile /etc/hyperledger/fabric-ca-server-config/ca.base.left-cert.pem --ca.keyfile /etc/hyperledger/fabric-ca-server-config/{replacehere}
```

Next we are going to start the network but before the set the following envrionment variables


```
export COMPOSE_PROJECT_NAME=net
export IMAGE_TAG=latest
export SYS_CHANNEL=base-sys-channel
```
 
 9. Now we are going to start network `docker-compose -f  docker-compose.yaml up -d`

 Once the build is completed you can check the active containers using `docker ps`

Next we are going entering the  `cli` container for the channel creation, peer joining and peer update operations.In the cli we are performing blockchain transaction as `org1` admin

10. Entering the cli container `docker exec -it cli bash`

Next we going to create the channel in order to do that we are going to use the channel artifact `channel.tx` which is been generated in  step 4 

11. Run the following to create the channel 
``` 
peer channel create -o orderer1.base.order:7050 -c basechannel  -f ./channel-artifacts/channel.tx --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/base.order/tlsca/tlsca.base.order-cert.pem
```

On successfull execution of the above command a `basechannel.block` file is returned.

12. Next we are going to join org1 peer to the channel using the following command `peer channel join -b basechannel.block`

13. Update the anchor peer for org1  
```
peer channel update \
	-o orderer1.base.order:7050 \
	-c basechannel \
	-f ./channel-artifacts/BaseLeftOrg.tx \
	--tls \
	--cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/base.order/orderers/orderer1.base.order/msp/tlscacerts/tlsca.base.order-cert.pem
  ```

We can use the same `basechannel.block` to join the org2 peer to the current channel but before that please set the following environment variables org2 peer 

```
CORE_PEER_MSPCONFIGPATH=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/base.right/users/Admin@base.right/msp
CORE_PEER_ADDRESS=peer1.base.right:9051
CORE_PEER_LOCALMSPID=RightOrgMSP
CORE_PEER_TLS_ROOTCERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/base.right/peers/peer1.base.right/tls/ca.crt
```


14. To join the org2 peer run `peer channel join -b basechannel.block`

15. Update the anchor peer for org2 
```
peer channel update \
	-o orderer1.base.order:7050 \
	-c basechannel \
	-f ./channel-artifacts/BaseRightOrg.tx \
	--tls \
	--cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/base.order/orderers/orderer1.base.order/msp/tlscacerts/tlsca.base.order-cert.pem
  ```

  Hurray! You have successfully created basechannel channel and made 2 organizations join to it.

