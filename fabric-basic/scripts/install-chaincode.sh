#!/bin/sh

echo '---PACKAGE CHAIN CODE---'

peer lifecycle chaincode package fabcar.tar.gz --path /opt/gopath/src/github.com/chaincode/go --lang golang --label fabcar_1

echo '---INSTALL CHAINCODE FOR ORG2---'

CHAINCODE_ID="$(peer lifecycle chaincode install fabcar.tar.gz)"

echo '---APPROVE CHAINCODE FOR ORG2----'

peer lifecycle chaincode approveformyorg  --channelID basechannel --name fabcar --version 1.0 --init-required --package-id $CHAINCODE_ID --sequence 1 -o orderer1.base.order:7050  --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/base.order/tlsca/tlsca.base.order-cert.pem


echo '---UPDATE ENVIRONMENT FOR ORG1-----'

 export CORE_PEER_ADDRESS=peer1.base.left:7051
 export CORE_PEER_LOCALMSPID=LeftOrgMSP
 export CORE_PEER_MSPCONFIGPATH=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/base.left/users/Admin@base.left/msp
 export CORE_PEER_TLS_ROOTCERT_FILE=/opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/base.left/peers/peer1.base.left/tls/ca.crt

echo '---INSTALL CHAINCODE FOR ORG1---'

sleep 5s

peer lifecycle chaincode install fabcar.tar.gz

 echo '---APPROVE CHAINCODE FOR ORG1----'

 peer lifecycle chaincode approveformyorg  --channelID basechannel --name fabcar --version 1.0 --init-required --package-id $CHAINCODE_ID --sequence 1 -o orderer1.base.order:7050  --tls --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/base.order/tlsca/tlsca.base.order-cert.pem

 echo '---COMMIT CHAINCODE---'

 peer lifecycle chaincode commit -o orderer1.base.order:7050 --channelID basechannel \
 --name fabcar --version 1.0 \
 --sequence 1 --init-required --tls true \
 --cafile /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/ordererOrganizations/base.order/tlsca/tlsca.base.order-cert.pem \
 --peerAddresses peer1.base.right:9051 \
 --tlsRootCertFiles /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/base.right/peers/peer1.base.right/tls/ca.crt \
 --peerAddresses peer1.base.left:7051  \
 --tlsRootCertFiles   /opt/gopath/src/github.com/hyperledger/fabric/peer/crypto/peerOrganizations/base.left/peers/peer1.base.left/tls/ca.crt