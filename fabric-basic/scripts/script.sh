#!/bin/sh

#Create the channel

echo '---CREATING CHANNEL---'

sleep 10s

peer channel create -o orderer1.base.order:7050 -c basechannel  -f /opt/gopath/src/github.com/hyperledger/fabric/peer/channel-artifacts/channel.tx --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/base.order/tlsca/tlsca.base.order-cert.pem

echo '---CHANNEL CREATED---'

echo 'JOIN ORG1:PEER TO CHANNEL'
sleep 10s
#join org1 peer to channel
peer channel join -b basechannel.block

echo '--UPDATE PEERS  FOR ORG:1---'

peer channel update \
	-o orderer1.base.order:7050 \
	-c basechannel \
	-f ./channel-artifacts/BaseLeftOrg.tx \
	--tls \
	--cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/base.order/orderers/orderer1.base.order/msp/tlscacerts/tlsca.base.order-cert.pem


echo '----UPDATE ENVIRONMENT FOR  ORG-2----'

export CORE_PEER_MSPCONFIGPATH=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/base.right/users/Admin@base.right/msp
export CORE_PEER_ADDRESS=peer1.base.right:9051
export CORE_PEER_LOCALMSPID=RightOrgMSP
export CORE_PEER_TLS_ROOTCERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/base.right/peers/peer1.base.right/tls/ca.crt


echo '----JOIN ORG-2:PEER TO THE CHANNEL-----'

sleep 10s

peer channel join -b basechannel.block

echo '----UPDATE THE ANCHOR PEER FOR ORG2-----'

peer channel update \
	-o orderer1.base.order:7050 \
	-c basechannel \
	-f ./channel-artifacts/BaseRightOrg.tx \
	--tls \
	--cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/base.order/orderers/orderer1.base.order/msp/tlscacerts/tlsca.base.order-cert.pem

exit;